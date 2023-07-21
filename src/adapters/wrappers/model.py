import importlib
import os
from typing import Any, Optional, Type, Union

from transformers import PreTrainedModel
from transformers.models.auto.auto_factory import getattribute_from_module
from transformers.models.auto.configuration_auto import model_type_to_module_name

from ..mixins import MODEL_MIXIN_MAPPING
from ..model_mixin import EmbeddingAdaptersWrapperMixin, ModelAdaptersMixin, ModelWithHeadsAdaptersMixin


SPECIAL_MODEL_TYPE_TO_MODULE_NAME = {
    "clip_vision_model": "clip",
    "clip_text_model": "clip",
}


def get_module_name(model_type: str) -> str:
    if model_type in SPECIAL_MODEL_TYPE_TO_MODULE_NAME:
        return SPECIAL_MODEL_TYPE_TO_MODULE_NAME[model_type]
    return model_type_to_module_name(model_type)


def wrap_model(model: PreTrainedModel) -> PreTrainedModel:
    if isinstance(model, ModelAdaptersMixin):
        return model

    # First, replace original module classes with their adapters counterparts
    model_name = get_module_name(model.config.model_type)
    modules_with_adapters = importlib.import_module(f".{model_name}.modeling_{model_name}", "adapters.models")
    for module in model.modules():
        # Check if module is a base model class
        if module.__class__.__name__ in MODEL_MIXIN_MAPPING:
            # Create new wrapper model class
            model_class = type(
                module.__class__.__name__, (MODEL_MIXIN_MAPPING[module.__class__.__name__], module.__class__), {}
            )
            module.__class__ = model_class
        elif module.__class__.__module__.startswith("transformers.models"):
            try:
                module_class = getattribute_from_module(
                    modules_with_adapters, module.__class__.__name__ + "WithAdapters"
                )
                module.__class__ = module_class
            except ValueError:
                # Silently fail and keep original module class
                pass

    # Next, check if model class itself is not replaced and has an adapter-supporting base class
    if not isinstance(model, ModelAdaptersMixin):
        if hasattr(model, "base_model_prefix") and hasattr(model, model.base_model_prefix):
            base_model = getattr(model, model.base_model_prefix)
            if isinstance(base_model, ModelAdaptersMixin):
                # Create new wrapper model class
                model_class_name = model.__class__.__name__
                model_class = type(
                    model_class_name, (EmbeddingAdaptersWrapperMixin, ModelWithHeadsAdaptersMixin, model.__class__), {}
                )
                model.__class__ = model_class

    # Finally, initialize adapters
    model.init_adapters(model.config)

    return model


def load_model(
    model_name_or_path: Optional[Union[str, os.PathLike]],
    model_class: Type[PreTrainedModel],
    *model_args: Any,
    **kwargs: Any
) -> PreTrainedModel:
    """
    Loads a pretrained model with adapters from the given path or url.

    Parameters:
        model_name_or_path (`str` or `os.PathLike`, *optional*):
            Parameter identical to PreTrainedModel.from_pretrained
        model_class (`PreTrainedModel` or `AutoModel`):
            The model class to load (e.g. EncoderDecoderModel and EncoderDecoderAdapterModel both work)
        model_args (sequence of positional arguments, *optional*):
            All remaining positional arguments will be passed to the underlying model's `__init__` method.
        kwargs (remaining dictionary of keyword arguments, *optional*):
            Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
            `output_attentions=True`).
    Returns:
        `PreTrainedModel`: The model with adapters loaded from the given path or url.
    """

    old_init = model_class.__init__

    def new_init(self, config, *args, **kwargs):
        old_init(self, config, *args, **kwargs)
        wrap_model(self)

    # wrap model after it is initialized but before the weights are loaded
    model_class.__init__ = new_init
    model = model_class.from_pretrained(model_name_or_path, *model_args, **kwargs)

    # restore original __init__ function for when other models of the same type are created
    model_class.__init__ = old_init

    return model