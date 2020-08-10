from transformers.pipelines import *
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler

from typing import List
import itertools
from tqdm import tqdm
from checklist_modeling_bert import ChecklistBertForQuestionAnswering


class ChecklistQuestionAnsweringPipeline(QuestionAnsweringPipeline):
    default_input_names = "question,context"

    def __init__(
            self,
            model: Union["PreTrainedModel", "TFPreTrainedModel"],
            tokenizer: PreTrainedTokenizer,
            modelcard: Optional[ModelCard] = None,
            framework: Optional[str] = None,
            task: str = "",
            device_ids=[0],
            device=torch.device("cuda"),
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            framework=framework,
            device=device_ids[0],
            task=task,
        )

        self.model.to(device)

        self.check_model_type(
            TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING if self.framework == "tf" else MODEL_FOR_QUESTION_ANSWERING_MAPPING
        )

        self.device_ids = device_ids
        if device_ids and len(device_ids) > 1:
            self.model = nn.DataParallel(model)

        self.device_type = device



    @staticmethod
    def create_sample(
            question: Union[str, List[str]], context: Union[str, List[str]]
    ) -> Union[SquadExample, List[SquadExample]]:
        """
        QuestionAnsweringPipeline leverages the SquadExample/SquadFeatures internally.
        This helper method encapsulate all the logic for converting question(s) and context(s) to SquadExample(s).
        We currently support extractive question answering.
        Arguments:
             question: (str, List[str]) The question to be ask for the associated context
             context: (str, List[str]) The context in which we will look for the answer.

        Returns:
            SquadExample initialized with the corresponding question and context.
        """
        if isinstance(question, list):
            return [SquadExample(None, q, c, None, None, None) for q, c in zip(question, context)]
        else:
            return SquadExample(None, question, context, None, None, None)



    def __call__(self, *args, **kwargs):
        """
        Args:
            We support multiple use-cases, the following are exclusive:
            X: sequence of SquadExample
            data: sequence of SquadExample
            question: (str, List[str]), batch of question(s) to map along with context
            context: (str, List[str]), batch of context(s) associated with the provided question keyword argument
        Returns:
            dict: {'answer': str, 'score": float, 'start": int, "end": int}
            answer: the textual answer in the intial context
            score: the score the current answer scored for the model
            start: the character index in the original string corresponding to the beginning of the answer' span
            end: the character index in the original string corresponding to the ending of the answer' span
        """
        # Set defaults values
        kwargs.setdefault("topk", 1)
        kwargs.setdefault("doc_stride", 128)
        kwargs.setdefault("max_answer_len", 16)
        kwargs.setdefault("max_seq_len", 32)
        kwargs.setdefault("max_question_len", 16)
        kwargs.setdefault("handle_impossible_answer", False)

        bsz = kwargs.pop("batch_size", 128)

        if kwargs["topk"] < 1:
            raise ValueError("topk parameter should be >= 1 (got {})".format(kwargs["topk"]))

        if kwargs["max_answer_len"] < 1:
            raise ValueError("max_answer_len parameter should be >= 1 (got {})".format(kwargs["max_answer_len"]))

        # Convert inputs to features
        examples = self._args_parser(*args, **kwargs)
        features_list, dataset = squad_convert_examples_to_features(
                examples=examples,
                tokenizer=self.tokenizer,
                max_seq_length=kwargs["max_seq_len"],
                doc_stride=kwargs["doc_stride"],
                max_query_length=kwargs["max_question_len"],
                # padding_strategy=PaddingStrategy.DO_NOT_PAD.value,
                is_training=False,
                tqdm_enabled=False,
                return_dataset='pt',
            )

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=bsz)


        all_answers = []
        for batch, example in tqdm(zip(eval_dataloader, examples), desc='Evaluating on checklist task',
                                   position=0, leave=True):
            # model_input_names = self.tokenizer.model_input_names + ["input_ids"]
            # fw_args = {k: [feature.__dict__[k] for feature in features] for k in model_input_names}

            batch = tuple(t.to(self.device_type) for t in batch)


            fw_args = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                # "token_type_ids": batch[2],
            }

            # Manage tensor allocation on correct device
            with self.device_placement():
                with torch.no_grad():
                    # Retrieve the score for the context tokens only (removing question tokens)
                    # fw_args = {k: torch.tensor(v).to(self.device_type) for (k, v) in fw_args.items()}
                    if isinstance(self.model, torch.nn.DataParallel):
                        fw_args["return_tuple"] = True
                    start, end = self.model(**fw_args)[:2]
                    # start, end = start.cpu().numpy(), end.cpu().numpy()

            min_null_score = 1000000  # large and positive
            answers = []
            # for (feature, start_, end_) in zip(features, start, end):
            # Ensure padded tokens & question tokens cannot belong to the set of candidate answers.
            # undesired_tokens = np.abs(np.array(feature.p_mask) - 1) & feature.attention_mask
            # undesired_tokens = np.abs(np.array(batch[5]) - 1) & batch[1]
            undesired_tokens = torch.abs(batch[5].int() - 1) & batch[1]

            # Generate mask
            undesired_tokens_mask = undesired_tokens == 0.0

            # Make sure non-context indexes in the tensor cannot contribute to the softmax
            start_ = torch.where(undesired_tokens_mask, torch.tensor(-10000.0).to(self.device_type), start)
            end_ = torch.where(undesired_tokens_mask, torch.tensor(-10000.0).to(self.device_type), end)


            # Normalize logits and spans to retrieve the answer
            # start_ = np.exp(start_ - np.log(np.sum(np.exp(start_), axis=-1, keepdims=True)))
            # end_ = np.exp(end_ - np.log(np.sum(np.exp(end_), axis=-1, keepdims=True)))
            start_ = torch.exp(start_ - torch.log(torch.sum(torch.exp(start_), axis=-1, keepdims=True)))
            end_ = torch.exp(end_ - torch.log(torch.sum(torch.exp(end_), axis=-1, keepdims=True)))

            if kwargs["handle_impossible_answer"]:
                min_null_score = min(min_null_score, (start_[0] * end_[0]).item())

            # Mask CLS
            start_[:, 0] = end_[:, 0] = 0.0

            starts, ends, scores = self.decode(start_, end_, kwargs["topk"], kwargs["max_answer_len"])
            feature_indices = batch[3]

            for i, feature_index in enumerate(feature_indices):
                eval_feature = features_list[feature_index.item()]
                eval_example = examples[feature_index.item()]
                char_to_word = np.array(eval_example.char_to_word_offset)

                # Convert the answer (tokens) back to the original text
                answers += [
                    {
                        "score": scores[i].item(),
                        "start": np.where(char_to_word == eval_feature.token_to_orig_map[starts[i].item()])[0][0].item(),
                        "end": np.where(char_to_word == eval_feature.token_to_orig_map[ends[i].item()])[0][-1].item(),
                        "answer": " ".join(
                            eval_example.doc_tokens[eval_feature.token_to_orig_map[starts[i].item()]: eval_feature.token_to_orig_map[ends[i].item()] + 1]
                        ),
                    }
                ]

            if kwargs["handle_impossible_answer"]:
                answers.append({"score": min_null_score, "start": 0, "end": 0, "answer": ""})

            # answers = sorted(answers, key=lambda x: x["score"], reverse=True)[: kwargs["topk"]]
            all_answers += answers

        if len(all_answers) == 1:
            return all_answers[0]
        return all_answers




    def decode(self, start: np.ndarray, end: np.ndarray, topk: int, max_answer_len: int) -> Tuple:
        """
        Take the output of any QuestionAnswering head and will generate probalities for each span to be
        the actual answer.
        In addition, it filters out some unwanted/impossible cases like answer len being greater than
        max_answer_len or answer end position being before the starting position.
        The method supports output the k-best answer through the topk argument.

        Args:
            start: numpy array, holding individual start probabilities for each token
            end: numpy array, holding individual end probabilities for each token
            topk: int, indicates how many possible answer span(s) to extract from the model's output
            max_answer_len: int, maximum size of the answer to extract from the model's output
        """
        # Ensure we have batch axis
        if start.ndim == 1:
            start = start[None]

        if end.ndim == 1:
            end = end[None]

        # Compute the score of each tuple(start, end) to be the real answer
        # outer = np.matmul(np.expand_dims(start, -1), np.expand_dims(end, 1))
        outer = torch.matmul(start.unsqueeze(-1), end.unsqueeze(1))

        # Remove candidate with end < start and end - start > max_answer_len
        # candidates = np.tril(np.triu(outer), max_answer_len - 1)
        candidates = torch.tril(torch.triu(outer), max_answer_len - 1)

        #  Inspired by Chen & al. (https://github.com/facebookresearch/DrQA)
        scores_flat = candidates.view(candidates.size(0), -1) #.flatten()
        # if topk == 1:
        # idx_sort = [np.argmax(scores_flat)]
        idx_sort = torch.argmax(scores_flat, dim=-1)
        # elif len(scores_flat) < topk:
        #     idx_sort = np.argsort(-scores_flat)
        # else:
        #     idx = np.argpartition(-scores_flat, topk)[0:topk]
        #     idx_sort = idx[np.argsort(-scores_flat[idx])]

        # start, end = np.unravel_index(idx_sort, candidates.shape)[1:]

        def torch_unravel_index(index, shape):
            out = []
            for dim in reversed(shape):
                out.append(index % dim)
                index = index // dim
            return tuple(reversed(out))


        start, end = torch_unravel_index(idx_sort, candidates.shape)[1:]
        red_cand = []
        for b in range(candidates.shape[0]):
            red_cand.append(candidates[b, start[b], end[b]].unsqueeze(0))
        return start, end, torch.cat(red_cand)




class ChecklistQuestionAnsweringPipelineFineGraned(ChecklistQuestionAnsweringPipeline):
    default_input_names = "question,context"

    def __init__(
            self,
            model: Union["PreTrainedModel", "TFPreTrainedModel"],
            tokenizer: PreTrainedTokenizer,
            modelcard: Optional[ModelCard] = None,
            framework: Optional[str] = None,
            task: str = "",
            device_ids=[0],
            device=torch.cuda,
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            framework=framework,
            device_ids=device_ids,
            task=task,
            device=device,
        )


    def __call__(self, *args, **kwargs):
        """
        Args:
            We support multiple use-cases, the following are exclusive:
            X: sequence of SquadExample
            data: sequence of SquadExample
            question: (str, List[str]), batch of question(s) to map along with context
            context: (str, List[str]), batch of context(s) associated with the provided question keyword argument
        Returns:
            dict: {'answer': str, 'score": float, 'start": int, "end": int}
            answer: the textual answer in the intial context
            score: the score the current answer scored for the model
            start: the character index in the original string corresponding to the beginning of the answer' span
            end: the character index in the original string corresponding to the ending of the answer' span
        """
        # Set defaults values
        kwargs.setdefault("topk", 1)
        kwargs.setdefault("doc_stride", 128)
        kwargs.setdefault("max_answer_len", 16)
        kwargs.setdefault("max_seq_len", 32)
        kwargs.setdefault("max_question_len", 16)
        kwargs.setdefault("handle_impossible_answer", False)

        bsz = kwargs.pop("batch_size", 128)
        study_num_layer = kwargs.pop("study_num_layer", 999999999)
        study_num_ts = kwargs.pop("study_num_ts", 999999999)
        study_num_neurons = kwargs.pop("study_num_neurons", 999999999)

        if kwargs["topk"] < 1:
            raise ValueError("topk parameter should be >= 1 (got {})".format(kwargs["topk"]))

        if kwargs["max_answer_len"] < 1:
            raise ValueError("max_answer_len parameter should be >= 1 (got {})".format(kwargs["max_answer_len"]))

        # Convert inputs to features
        examples = self._args_parser(*args, **kwargs)
        features_list, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=self.tokenizer,
            max_seq_length=kwargs["max_seq_len"],
            doc_stride=kwargs["doc_stride"],
            max_query_length=kwargs["max_question_len"],
            # padding_strategy=PaddingStrategy.DO_NOT_PAD.value,
            is_training=False,
            tqdm_enabled=False,
            return_dataset='pt',
        )

        study_num_ts = min(max([sum(f.attention_mask) for f in features_list]), study_num_ts)

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=bsz)

        if isinstance(self.model, torch.nn.DataParallel):
            study_num_neurons = min(self.model.module.config.dim, study_num_neurons)
            study_num_layer = min(self.model.module.config.n_layers, study_num_layer)
        else:
            study_num_neurons = min(self.model.config.dim, study_num_neurons)
            study_num_layer = min(self.model.config.n_layers, study_num_layer)

        neuron_level_answers = {
            l: {ts: {n: [] for n in range(study_num_neurons)} for ts in range(study_num_ts)} for l in
            range(study_num_layer)}
        for batch, example in tqdm(zip(eval_dataloader, examples), desc='Evaluating on checklist task'):
            # model_input_names = self.tokenizer.model_input_names + ["input_ids"]
            # fw_args = {k: [feature.__dict__[k] for feature in features] for k in model_input_names}

            batch = tuple(t.to(self.device_type) for t in batch)
            reset_cache = True

            fw_args = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
            }

            feature_indices = batch[3]

            for layer, ts, neu in tqdm(itertools.product(range(study_num_layer - 1, -1, -1),
                                                    range(study_num_ts - 1, -1, -1),
                                                    range(study_num_neurons - 1, -1, -1)), desc='Iterating over all neurons',
                                                    position=0, leave=True):

                # neu_mask = torch.zeros((batch[0].size(0), kwargs["max_seq_len"], dim)).to(self.device_type)
                # neu_mask[:, ts, neu] = 1
                # neu_mask = neu_mask.bool()
                reset_cache = False

                fw_args['neu_mask_layer'] = layer
                fw_args['mask_ts'] = ts
                fw_args['mask_neu'] = neu
                fw_args['reset_cache'] = reset_cache

                # Manage tensor allocation on correct device
                with self.device_placement():
                    with torch.no_grad():
                        # Retrieve the score for the context tokens only (removing question tokens)
                        # fw_args = {k: torch.tensor(v).to(self.device_type) for (k, v) in fw_args.items()}
                        if isinstance(self.model, torch.nn.DataParallel):
                            fw_args["return_tuple"] = True
                        start, end = self.model(**fw_args)[:2]
                        # start, end = start.cpu().numpy(), end.cpu().numpy()

                min_null_score = 1000000  # large and positive
                # for (feature, start_, end_) in zip(features, start, end):
                # Ensure padded tokens & question tokens cannot belong to the set of candidate answers.
                # undesired_tokens = np.abs(np.array(feature.p_mask) - 1) & feature.attention_mask
                # undesired_tokens = np.abs(np.array(batch[5]) - 1) & batch[1]
                undesired_tokens = torch.abs(batch[5].int() - 1) & batch[1]

                # Generate mask
                undesired_tokens_mask = undesired_tokens == 0.0

                # Make sure non-context indexes in the tensor cannot contribute to the softmax
                start_ = torch.where(undesired_tokens_mask, torch.tensor(-10000.0).to(self.device_type), start)
                end_ = torch.where(undesired_tokens_mask, torch.tensor(-10000.0).to(self.device_type), end)

                # Normalize logits and spans to retrieve the answer
                # start_ = np.exp(start_ - np.log(np.sum(np.exp(start_), axis=-1, keepdims=True)))
                # end_ = np.exp(end_ - np.log(np.sum(np.exp(end_), axis=-1, keepdims=True)))
                start_ = torch.exp(start_ - torch.log(torch.sum(torch.exp(start_), axis=-1, keepdims=True)))
                end_ = torch.exp(end_ - torch.log(torch.sum(torch.exp(end_), axis=-1, keepdims=True)))

                if kwargs["handle_impossible_answer"]:
                    min_null_score = min(min_null_score, (start_[0] * end_[0]).item())

                # Mask CLS
                start_[:, 0] = end_[:, 0] = 0.0

                starts, ends, scores = self.decode(start_, end_, kwargs["topk"], kwargs["max_answer_len"])

                for i, feature_index in enumerate(feature_indices):
                    eval_feature = features_list[feature_index.item()]
                    eval_example = examples[feature_index.item()]
                    char_to_word = np.array(eval_example.char_to_word_offset)

                    # Convert the answer (tokens) back to the original text
                    neuron_level_answers[layer][ts][neu] += [
                        {
                            "score": scores[i].item(),
                            "start": np.where(char_to_word == eval_feature.token_to_orig_map[starts[i].item()])[0][
                                0].item(),
                            "end": np.where(char_to_word == eval_feature.token_to_orig_map[ends[i].item()])[0][
                                -1].item(),
                            "answer": " ".join(
                                eval_example.doc_tokens[
                                eval_feature.token_to_orig_map[starts[i].item()]: eval_feature.token_to_orig_map[
                                                                                      ends[i].item()] + 1]
                            ),
                        }
                    ]
        return neuron_level_answers





SUPPORTED_TASKS["checklist-question-answering"] = {"impl": ChecklistQuestionAnsweringPipelineFineGraned,
                                                    "pt": AutoModelForQuestionAnswering if is_torch_available() else None,
                                                            "default": {
                                                                "model": {"pt": "distilbert-base-cased-distilled-squad"},
                                                            }
                                                   }
                                                   # "pt":  ChecklistBertForQuestionAnswering,
                                                   # "default": {
                                                   #     "model": {"pt": "bert-base-uncased"}
                                                   #            }
                                                   # }



def pipeline(
    task: str,
    model: Optional = None,
    config: Optional[Union[str, PretrainedConfig]] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    framework: Optional[str] = None,
    **kwargs
) -> Pipeline:
    """
    Utility factory method to build a pipeline.

    Pipeline are made of:

        - A Tokenizer instance in charge of mapping raw textual input to token
        - A Model instance
        - Some (optional) post processing for enhancing model's output


    Args:
        task (:obj:`str`):
            The task defining which pipeline will be returned. Currently accepted tasks are:

            - "feature-extraction": will return a :class:`~transformers.FeatureExtractionPipeline`
            - "sentiment-analysis": will return a :class:`~transformers.TextClassificationPipeline`
            - "ner": will return a :class:`~transformers.TokenClassificationPipeline`
            - "question-answering": will return a :class:`~transformers.QuestionAnsweringPipeline`
            - "fill-mask": will return a :class:`~transformers.FillMaskPipeline`
            - "summarization": will return a :class:`~transformers.SummarizationPipeline`
            - "translation_xx_to_yy": will return a :class:`~transformers.TranslationPipeline`
            - "text-generation": will return a :class:`~transformers.TextGenerationPipeline`
        model (:obj:`str` or :obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`, `optional`, defaults to :obj:`None`):
            The model that will be used by the pipeline to make predictions. This can be :obj:`None`,
            a model identifier or an actual pre-trained model inheriting from
            :class:`~transformers.PreTrainedModel` for PyTorch and :class:`~transformers.TFPreTrainedModel` for
            TensorFlow.

            If :obj:`None`, the default for this pipeline will be loaded.
        config (:obj:`str` or :obj:`~transformers.PretrainedConfig`, `optional`, defaults to :obj:`None`):
            The configuration that will be used by the pipeline to instantiate the model. This can be :obj:`None`,
            a model identifier or an actual pre-trained model configuration inheriting from
            :class:`~transformers.PretrainedConfig`.

            If :obj:`None`, the default for this pipeline will be loaded.
        tokenizer (:obj:`str` or :obj:`~transformers.PreTrainedTokenizer`, `optional`, defaults to :obj:`None`):
            The tokenizer that will be used by the pipeline to encode data for the model. This can be :obj:`None`,
            a model identifier or an actual pre-trained tokenizer inheriting from
            :class:`~transformers.PreTrainedTokenizer`.

            If :obj:`None`, the default for this pipeline will be loaded.
        framework (:obj:`str`, `optional`, defaults to :obj:`None`):
            The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified
            and both frameworks are installed, will default to PyTorch.

    Returns:
        :class:`~transformers.Pipeline`: Class inheriting from :class:`~transformers.Pipeline`, according to
        the task.

    Examples::

        from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

        # Sentiment analysis pipeline
        pipeline('sentiment-analysis')

        # Question answering pipeline, specifying the checkpoint identifier
        pipeline('question-answering', model='distilbert-base-cased-distilled-squad', tokenizer='bert-base-cased')

        # Named entity recognition pipeline, passing in a specific model and tokenizer
        model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        pipeline('ner', model=model, tokenizer=tokenizer)
    """

    # Retrieve the task
    if task not in SUPPORTED_TASKS:
        raise KeyError("Unknown task {}, available tasks are {}".format(task, list(SUPPORTED_TASKS.keys())))

    framework = framework or get_framework(model)

    targeted_task = SUPPORTED_TASKS[task]
    task_class, model_class = targeted_task["impl"], targeted_task[framework]

    # Use default model/config/tokenizer for the task if no model is provided
    if model is None:
        model = targeted_task["default"]["model"][framework]

    # Try to infer tokenizer from model or config name (if provided as str)
    if tokenizer is None:
        if isinstance(model, str):
            tokenizer = model
        elif isinstance(config, str):
            tokenizer = config
        else:
            # Impossible to guest what is the right tokenizer here
            raise Exception(
                "Impossible to guess which tokenizer to use. "
                "Please provided a PretrainedTokenizer class or a path/identifier to a pretrained tokenizer."
            )

    modelcard = None
    # Try to infer modelcard from model or config name (if provided as str)
    if isinstance(model, str):
        modelcard = model
    elif isinstance(config, str):
        modelcard = config

    # Instantiate tokenizer if needed
    if isinstance(tokenizer, (str, tuple)):
        if isinstance(tokenizer, tuple):
            # For tuple we have (tokenizer name, {kwargs})
            tokenizer = AutoTokenizer.from_pretrained(tokenizer[0], **tokenizer[1])
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    # Instantiate config if needed
    if isinstance(config, str):
        config = AutoConfig.from_pretrained(config)

    # Instantiate modelcard if needed
    if isinstance(modelcard, str):
        modelcard = ModelCard.from_pretrained(modelcard)

    # Instantiate model if needed
    if isinstance(model, str):
        # Handle transparent TF/PT model conversion
        model_kwargs = {}
        if framework == "pt" and model.endswith(".h5"):
            model_kwargs["from_tf"] = True
            logger.warning(
                "Model might be a TensorFlow model (ending with `.h5`) but TensorFlow is not available. "
                "Trying to load the model with PyTorch."
            )
        elif framework == "tf" and model.endswith(".bin"):
            model_kwargs["from_pt"] = True
            logger.warning(
                "Model might be a PyTorch model (ending with `.bin`) but PyTorch is not available. "
                "Trying to load the model with Tensorflow."
            )
        model = model_class.from_pretrained(model, config=config, **model_kwargs)

    return task_class(model=model, tokenizer=tokenizer, modelcard=modelcard, framework=framework, task=task, **kwargs)






