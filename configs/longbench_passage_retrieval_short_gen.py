from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import (
    LongBenchRetrievalEvaluator,
    LongBenchpassage_retrieval_enDataset,
)

# LongBench passage_retrieval_en restricted to samples that fit LLaDA's 4096-token
# context, so a wrong eviction shows up as a wrong paragraph rather than as a
# needle that truncation removed.
LongBench_passage_retrieval_short_reader_cfg = dict(
    input_columns=['context', 'input'],
    output_column='answers',
    train_split='test',
    test_split='test',
)

LongBench_passage_retrieval_short_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\nThe answer is: ',
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=32),
)

LongBench_passage_retrieval_short_eval_cfg = dict(
    evaluator=dict(type=LongBenchRetrievalEvaluator), pred_role='BOT'
)

LongBench_passage_retrieval_short_datasets = [
    dict(
        type=LongBenchpassage_retrieval_enDataset,
        abbr='LongBench_passage_retrieval_short',
        path='opencompass/Longbench',
        name='passage_retrieval_short',
        reader_cfg=LongBench_passage_retrieval_short_reader_cfg,
        infer_cfg=LongBench_passage_retrieval_short_infer_cfg,
        eval_cfg=LongBench_passage_retrieval_short_eval_cfg,
    )
]
