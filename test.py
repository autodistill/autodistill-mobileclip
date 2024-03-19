from autodistill_mobileclip import MobileCLIP
from autodistill.detection import CaptionOntology

# define an ontology to map class names to our MobileCLIP prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = MobileCLIP(
    ontology=CaptionOntology(
        {
            "person": "person",
            "a forklift": "forklift",
            "coffee": "coffee"
        }
    )
)
result = base_model.predict("IMG_2584.jpeg")

print(result)