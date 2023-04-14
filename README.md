|      similairty     |            evaluator               |   tp   |  fp    | miss   |  
| ------------------- | ---------------------------------- | ------ | ------ | ------ |
| albert-small (0.95) | distance(max_distance=4.0, False)  |   170  |   0    |   63   |
| albert-small (0.5)  | distance(max_distance=4.0, False)  |   227  |   6    |   0    |
| albert-small (0.5 ) | onnx                               |   180  |   3    |   50   |
| albert-small (0.5 ) | kreprocical(topk=2)                |   218  |   1    |   14   |

