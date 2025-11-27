# Description

Call ./deploy.sh to deploy the app to modal.

You will need to add your modal account ID into the URL in hit_endpoint.py. 

## Serving requests
`request` contains code for generating and running example tests with random batches, on dev data.
Running a test will create a `simulation_summary.json` inside the test specific folder where the `"results"` field will be a list with items that look like this (one per batch, where a batch can have one or more requests)
```
    {
      "batch_id": 2,
      "batch_size": 1,
      "scheduled_arrival_time": 33.35654626306513,
      "actual_send_time": 33.37366533279419,
      "request_duration": 14.475500345230103,
      "completion_time": 47.84920930862427,
      "status_code": 200,
      "prompt_idxs": [
        5
      ],
      "response": {
        "choices": [
          {
            "text": "The answer is (A).",
            "full_text": "user\nThe following is a multiple choice question (with answers) about  college medicine.  Output the answer in the format of \"The answer is (X)\" at the end.\n\nQuestion: A teacher sets up a reward system for her elementary school students. At the end of each day, she gives a sticker to each student who showed up on time that morning. At the end of each week, she gives a sticker to any student who got above a 90% on three quizzes in a row. After months of this regimen, she finds that performance on the quizzes has increased significantly but that tardiness has only decreased slightly. Which of the following best explains the teacher\u2019s observation?\n Options:\nA. Variable ratio schedules create the strongest responses and behavior that is the least susceptible to extinction.\nB. The students had more intrinsic motivation to do well on quizzes than to show up on time.\nC. The students\u2019 behavior change was stronger in response to a fixed-ratio schedule than it was to a continuous reinforcement schedule.\nD. The students\u2019 behavior change was stronger in response to a fixed-ratio schedule than it was to a variable-interval schedule.\nAnswer:\nassistant\n<think>\n\n</think>\n\nThe answer is (A).",
            "index": 0,
            "finish_reason": "stop"
          }
        ],
        "model": "[andrewid]-system-[system_number]",
        "usage": {
          "prompt_tokens": 245,
          "completion_tokens": 7,
          "total_tokens": 252
        }
      },
      "error": null
    },
```
**Note that the `"prompt_idxs"` field is what corresponds to the *original indices of the prompts in a batch*** where the prompt are sampled from a list. The `"index"` field corresponds to the index of a particular example wrt a given batch -- currently overloaded with a field name in the eval script.

We have not released code for making the aggregated result json directly compatible with the eval script, but it should be a relatively simple transformation. At final test time, we will perform this transformation for you.

## Evaluating outputs
Evaluation scripts are run over json outputs where task mappings are visible on our end. There are a couple relevant discrepancies between expected formats and what you may already be used to:

1. **One key discrepancy with the starter example contained in the `example_code` folder is that we expect the outputs to contain the generated text only, without the input prompt.** You may already be doing this -- it is just worth noting that the system in `starter_example.py` happens to output the input prompt as well (which you should not do!) Concretely, with respect to the example above, we will use the `"text"` field and *not* the `"full_text"` field, so your systems should output `"text"` fields that contain only the generated text (you do not need to also return a separate `"full_text"` field).
2. In the original grading for the graph task, we were calling a `submit_paths` function in the evaluator. We have assumed that the final outputs will basically be the same in students' submitted systems, but now we are extracting paths and weights after `submit_paths` from the generations. We are also allowing an alternative where if a system directly outputs the correct dict with correct paths and weights, it is also graded. Similar to the original eval script, if the number of shortest paths asks for more than one path, and say a system gets 2/3 paths right, they will be give a score of .67 and not 0.

