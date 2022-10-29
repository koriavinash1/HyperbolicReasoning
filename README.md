# Hierarchical Symbolic Reasoning in Hyperbolic Space for Deep Discriminative Models


## Installation 

```
cd SymbolicReasoning
pip3 install -r requirements.py
```


## Steps

> Pre-training Classifiers

```
cd Classifiers

chmod +x train.sh

# change data and logs directory in train.sh
./train.sh 
```

> Training Explainer Model

```
# change data and logs directory in main.py (in function arguments)
# load the pretrained classifiers in main ref. line 74-95

python main.py
```

> Generate Explanations

```
# Follow Inference notebook in playground to generate explanations
```
