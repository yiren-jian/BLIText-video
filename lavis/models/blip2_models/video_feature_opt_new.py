"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_opt import OPTForCausalLM, OPTConfig
from transformers import AutoTokenizer

from transformers import BertTokenizer
from lavis.models.blip2_models.modeling_bert import BertConfig, BertModel


@registry.register_model("video_feature_opt_new")
class VideoFeatureOPTNew(Blip2Base):
    """
    BLIP2 OPT model.
    Supported model types:
        - pretrained_opt2.7b: pretrained model with OPT2.7b
        - pretrained_opt6.7b: pretrained model with OPT6.7b
        - caption_coco_opt2.7b: fintuned image captioning model with OPT2.7b
        - caption_coco_opt6.7b: fintuned image captioning model with OPT6.7b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_opt", "caption_coco_opt2.7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        # "pretrain_opt2.7b": "configs/models/blip2/blip2_pretrain_opt2.7b.yaml",
        # "pretrain_opt6.7b": "configs/models/blip2/blip2_pretrain_opt6.7b.yaml",
        "caption_coco_opt2.7b": "configs/models/blip2/blip2_caption_opt2.7b.yaml",
        "caption_coco_opt6.7b": "configs/models/blip2/blip2_caption_opt6.7b.yaml",
    }

    def __init__(
        self,
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        opt_model="facebook/opt-2.7b",
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False,
    ):
        super().__init__()

        ###### New code starts
        self.I3D_hdim = 1024
        self.VL_adaptor = BertModel.from_pretrained("bert-base-uncased")
        self.VL_adaptor.feat_proj = nn.Linear(self.I3D_hdim, self.VL_adaptor.config.hidden_size)   # 1024 --> 768
        for name, param in self.VL_adaptor.embeddings.named_parameters():
            param.requires_grad = False    #### not used during training
        ###### New code ends

        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
        self.opt_model = OPTForCausalLM.from_pretrained(
            opt_model, torch_dtype=torch.float16
        )
        for name, param in self.opt_model.named_parameters():
            param.requires_grad = False
        self.eos_token_id = self.opt_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]

        self.opt_proj = nn.Linear(
            self.VL_adaptor.config.hidden_size, self.opt_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

        ###### new code starts #####
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.Darkformer = BertModel.from_pretrained("bert-base-uncased")
        self.Darkformer.cls_proj = nn.Linear(self.Darkformer.config.hidden_size, num_query_token*self.Darkformer.config.hidden_size)
        self.Darkformer.pooler = nn.Sequential(nn.Linear(self.Darkformer.config.hidden_size, self.Darkformer.config.hidden_size), nn.Tanh())
        self.Darkformer.opt_proj = nn.Linear(
            self.Darkformer.config.hidden_size, self.opt_model.config.hidden_size
        )
        for name, param in self.Darkformer.named_parameters():
            param.requires_grad = False
        self.Darkformer.eval()

        self.loss_weight = 100.0
        ###### new code ends #####

    def forward(self, samples):
        image = samples["image"]    ### video I3D featrues bsz x 32 x 1024
        image_mask = samples["mask"]
        image_embeds = self.VL_adaptor.feat_proj(image)   ### bsz x 32 x 768
        image_output = self.VL_adaptor.forward(attention_mask=image_mask, inputs_embeds=image_embeds)   ### bsz x 32 x 768

        inputs_opt = self.opt_proj(image_output.last_hidden_state)   ### bsz x 32 x 2560
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(image.device)

        self.opt_tokenizer.padding_side = "right"

        text = [t + "\n" for t in samples["text_input"]]

        ###### new code starts ######
        # with torch.no_grad():
        #     text_tokens = self.tokenizer(
        #         text,
        #         padding="max_length",
        #         truncation=True,
        #         max_length=self.max_txt_len,
        #         return_tensors="pt",
        #     ).to(image.device)
        #
        #     text_output = self.Darkformer.forward(
        #         text_tokens.input_ids,
        #         attention_mask=text_tokens.attention_mask,
        #         return_dict=True,
        #     )
        #
        #     bsz, n_query, hdim = text_output.last_hidden_state.shape
        #     inputs_opt_ref = self.Darkformer.cls_proj(text_output.last_hidden_state[:, 0, :])
        #     inputs_opt_ref = inputs_opt_ref.reshape(bsz, n_query, hdim)
        #     inputs_opt_ref = self.Darkformer.pooler(inputs_opt_ref)
        #     inputs_opt_ref = self.Darkformer.opt_proj(inputs_opt_ref)
        #
        # inputs_opt_ref = inputs_opt_ref.detach()
        # loss_align = nn.MSELoss()(inputs_opt, inputs_opt_ref) * self.loss_weight
        ###### new code ends  ######

        opt_tokens = self.opt_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)

        targets = opt_tokens.input_ids.masked_fill(
            opt_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )
        if self.prompt:
            targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt

        empty_targets = (
            torch.ones(atts_opt.size(), dtype=torch.long).to(image.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

        outputs = self.opt_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss_lm = outputs.loss   ### change loss --> loss_lm

        loss = loss_lm

        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        image_mask = samples["mask"]
        with torch.cuda.amp.autocast(
            enabled=(self.device != torch.device("cpu"))
        ):
            # image = samples["image"]    ### video I3D featrues bsz x 32 x 1024
            image_embeds = self.VL_adaptor.feat_proj(image)   ### bsz x 32 x 768
            image_output = self.VL_adaptor.forward(attention_mask=image_mask, inputs_embeds=image_embeds)   ### bsz x 32 x 768

            inputs_opt = self.opt_proj(image_output.last_hidden_state)   ### bsz x 32 x 2560
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(image.device)

            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = self.prompt

            prompt = [prompt] * image.size(0)

            opt_tokens = self.opt_tokenizer(prompt, return_tensors="pt").to(image.device)
            input_ids = opt_tokens.input_ids
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

            if use_nucleus_sampling:
                query_embeds = inputs_opt.repeat_interleave(num_captions, dim=0)
                num_beams = 1
            else:
                query_embeds = inputs_opt.repeat_interleave(num_beams, dim=0)

            outputs = self.opt_model.generate(
                input_ids=input_ids,
                query_embeds=query_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

            prompt_length = opt_tokens.input_ids.shape[1]
            output_text = self.opt_tokenizer.batch_decode(
                outputs[:, prompt_length:], skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]
            return output_text

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):

        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        opt_model = cfg.get("opt_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        model = cls(
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            opt_model=opt_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
        )
        model.load_checkpoint_from_config(cfg)

        return model
