import numpy as np

def evaluate_disc(agent, val_dataset):
    expert_inputs = np.concatenate([val_dataset.expert_observations, val_dataset.expert_actions], axis=-1)
    random_inputs = np.concatenate([val_dataset.union_observations, val_dataset.union_actions], axis=-1)
    expert_labels = np.ones_like(expert_inputs.shape[0], dtype=np.bool)
    random_labels = np.zeros_like(random_inputs.shape[0], dtype=np.bool)
    
    expert_logits = agent.cost(expert_inputs)
    random_logits = agent.cost(random_inputs)
    
    expert_acc = ((expert_logits>0)==expert_labels).mean()
    random_acc = ((random_logits>0)==random_labels).mean()
    
    return expert_acc.item(), random_acc.item()