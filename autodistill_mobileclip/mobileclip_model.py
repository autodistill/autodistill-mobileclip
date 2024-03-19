import os
from dataclasses import dataclass

import torch
import sys
from typing import Union

import numpy as np
import supervision as sv
from autodistill.classification import ClassificationBaseModel
from autodistill.core.embedding_model import EmbeddingModel
from autodistill.core.embedding_ontology import EmbeddingOntology, compare_embeddings
from autodistill.detection import CaptionOntology
from autodistill.helpers import load_image

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINTS = {
    "s0": "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s0.pt"
}

@dataclass
class MobileCLIP(ClassificationBaseModel, EmbeddingModel):
    ontology: Union[EmbeddingOntology, CaptionOntology]

    def __init__(self, ontology: Union[EmbeddingOntology, CaptionOntology], checkpoint: str = "s0"):
        self.ontology = ontology

        if not os.path.exists(f"{HOME}/.cache/autodistill/mobileclip"):
            os.makedirs(f"{HOME}/.cache/autodistill/mobileclip")

            os.system(
                f"cd {HOME}/.cache/autodistill/mobileclip && git clone https://github.com/apple/ml-mobileclip.git && cd ml-mobileclip && pip install -e ."
            )
            # pip install e
        
        # wget checkpoint
        if not os.path.exists(f"{HOME}/.cache/autodistill/mobileclip/{checkpoint}.pt"):
            os.system(
                f"wget {CHECKPOINTS[checkpoint]} -O {HOME}/.cache/autodistill/mobileclip/{checkpoint}.pt"
            )

        # add clip path to path
        sys.path.insert(0, f"{HOME}/.cache/autodistill/mobileclip/ml-mobileclip")

        import mobileclip

        model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_s0', pretrained=f"{HOME}/.cache/autodistill/mobileclip/{checkpoint}.pt")
        tokenizer = mobileclip.get_tokenizer('mobileclip_s0')

        self.clip_model = model
        self.clip_preprocess = preprocess
        self.tokenizer = tokenizer

        # if Ontology is EmbeddingOntologyImage, then run process
        if isinstance(self.ontology, EmbeddingOntology):
            self.ontology.process(self)

        # get ontology class name
        self.ontology_type = self.ontology.__class__.__name__
        self.labels = self.tokenizer(self.ontology.prompts())

    def embed_image(self, input: str) -> np.ndarray:
        image = load_image(input, return_format="PIL")
        image = self.clip_preprocess(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)

        return image_features.cpu().numpy()

    def embed_text(self, input: str) -> np.ndarray:
        return (
            self.clip_model.encode_text(self.tokenize([input]).to(DEVICE)).cpu().numpy()
        )

    def predict(self, input: str) -> sv.Classifications:
        image = load_image(input, return_format="PIL")
        image = self.clip_preprocess(image).unsqueeze(0).to(DEVICE)

        if isinstance(self.ontology, EmbeddingOntology):
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image)

                return compare_embeddings(
                    image_features.cpu().numpy(), self.ontology.embeddingMap.values()
                )
        else:
            labels = self.ontology.prompts()

            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = self.clip_model.encode_image(image)
                text_features = self.clip_model.encode_text(self.tokenizer(labels).to(DEVICE))
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

                probs = text_probs[0].cpu().numpy()

            return sv.Classifications(
                class_id=np.array([i for i in range(len(labels))]),
                confidence=np.array(probs).flatten(),
            )