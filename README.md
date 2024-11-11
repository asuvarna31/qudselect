## QUDSELECT: Selective Decoding for Questions Under Discussion Parsing
## QUD Parser Training & Decoding
The files are under `single_joint_model/'.
### Setup
```
pip install -r requirements.txt
```
### Instruction data generation
```
python data/data_generation.py
```
This transforms the original train/val data into sentence-level instruction format.
### Finetuning
```
bash scripts/finetune_single_joint_lora_with_accelerate.sh
```
### Decoding
First decode multiple anchors for each answer sentence:
```
bash scripts/eval_single_joint_anchor.sh
```
Then prepare the question generation prompts based on the decoded anchors:
```
python data/prepare_question_pred_data.py
```
Decode multiple questions:
```
bash scripts/eval_single_joint_question.sh
```
Reformat the output:
```
python data/reformat_output.py
```
## Selective Decoding 
### Get Criteria Scores 
Run the following on the decoded anchors & questions:
```
python selective_decoding/rule_based_approaches.py
```
Then, run the follwoing to get the final selected quds:
```
python selective_decoding/get_final_quds.py
```

## Automatic Evaluator
The original data from QUDEVAL and oversampled data for training the supervised classifiers is in automatic_evaluators/data.

## Citation 
Please cite us if our paper inspired your work!

```bibtext
@misc{suvarna2024qudselectselectivedecodingquestions,
      title={QUDSELECT: Selective Decoding for Questions Under Discussion Parsing}, 
      author={Ashima Suvarna and Xiao Liu and Tanmay Parekh and Kai-Wei Chang and Nanyun Peng},
      year={2024},
      eprint={2408.01046},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.01046}, 
}
```
