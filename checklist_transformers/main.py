import argparse
from checklist_pipelines import *
import numpy as np
from checklist.test_suite import TestSuite
from batched_test_suite import BatchedTestSuite, BatchedMFT
import json
import torch.backends.cudnn as cudnn




parser = argparse.ArgumentParser(description='main script for checklist with transformers')
parser.add_argument('--device', default='cuda',
                    help='device assignment ("cpu" or "cuda")')
parser.add_argument('--device-ids', default=[0], type=int, nargs='+',
                    help='device ids assignment (e.g 0 1 2 3')
parser.add_argument('--seed', default=123, type=int,
                    help='random seed (default: 123)')
parser.add_argument('--batch-size', default=128, type=int,
                    help='batch size')
parser.add_argument('--num-draws', default=None, type=int,
                    help='number of samples to draws in checlist run function')
parser.add_argument('--task-type', default='Negation', type=str,
                    help='checklist task type')
parser.add_argument('--num-tasks', default=1, type=int,
                    help='number of tasks to perform from task-type')
parser.add_argument('--study-num-layer', default=999999999, type=int,
                    help='number of layers to perform fine-graned study on')
parser.add_argument('--study-num-ts', default=999999999, type=int,
                    help='number of timestamps to perform fine-graned study on')
parser.add_argument('--study-num-neurons', default=999999999, type=int,
                    help='number of neurons to perform fine-graned study on')




def load_squad(fold='validation'):
    answers = []
    data = []
    files = {
        'validation': '/media/drive/Datasets/squad/dev-v1.1.json',
        'train': '/media/drive/Datasets/squad/train-v1.1.json',
        }
    f = json.load(open(files[fold]))
    for t in f['data']:
        for p in t['paragraphs']:
            context = p['context']
            for qa in p['qas']:
                data.append({'passage': context, 'question': qa['question'], 'id': qa['id']})
                answers.append(set([(x['text'], x['answer_start']) for x in qa['answers']]))
    return data, answers


def format_squad_with_context(x, pred, conf, label=None, *args, **kwargs):
    c, q = x
    ret = 'C: %s\nQ: %s\n' % (c, q)
    if label is not None:
        ret += 'A: %s\n' % label
    ret += 'P: %s\n' % pred
    return ret


def main():
    args = parser.parse_args()
    main_worker(args)


def main_worker(args):
    torch.manual_seed(args.seed)

    logging.debug("run arguments: %s", args)

    # if 'cuda' in args.device and torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(args.seed)
    #     torch.cuda.set_device(args.device_ids[0])
    #     cudnn.benchmark = True
    #
    # else:
    #     args.device_ids = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_pipe = pipeline("checklist-question-answering", device_ids=args.device_ids, device=device)
    # model_pipe = pipeline("question-answering", device=0)


    suite_path = 'release_data/squad/squad_suite.pkl'
    suite = TestSuite.from_file(suite_path)
    suite = BatchedTestSuite(seed=args.seed, batch_size=args.batch_size, **suite.__dict__)

    filtered_tasks = [k for k in suite.test_ranges.keys() if args.task_type in k][: args.num_tasks]
    for k in suite.test_ranges.keys():
        if k not in filtered_tasks and k in suite.tests:
            suite.remove(k)
        else:
            suite.tests[k] = BatchedMFT(**suite.tests[k].__dict__)

    def predconfs(context_question_pairs):
        model_preds = model_pipe(question=context_question_pairs[1],
                                 context=context_question_pairs[0],
                                 truncation=True,
                                 batch_size=args.batch_size,
                                 study_num_layer=args.study_num_layer,
                                 study_num_ts=args.study_num_ts,
                                 study_num_neurons=args.study_num_neurons,
                                )
        return model_preds


    suite.run(predconfs, n=args.num_draws, overwrite=True)


if __name__ == '__main__':
    main()