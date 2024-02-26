from typing import List
import numpy as np

def bleu_score(predicted: List[int], target: List[int], N: int) -> float:
    """
    Finds the BLEU-N score between the predicted sentence and a single reference (target) sentence.
    Feel free to deviate from this skeleton if you prefer.

    Edge case: If the length of the predicted sentence or the target is less than N, return 0.
    """
    if len(predicted) < N or len(target) < N:
        # TODO
        return 0 
    
    def C(y, g):
        # TODO how many times does n-gram g appear in y?
        # ***for each gram in n-grams, how many time the gram appear in sequence***
        n = len(g)
        count = 0 
        left, right = 0, n
        # sliding window
        while right <= len(y):
            if g == tuple(y[left:right]):

                count += 1
            left += 1
            right += 1
        return count
                        

    geo_mean = 1
    for n in range(1, N+1): # for all N grams (e.g. N = 3, 4, 5, 6, ..., n)
        grams = set() # unique n-grams
        Cs = []
        for i in range(len(predicted)-n+1):
            # TODO add to grams
            pred_gram = tuple(predicted[i:i+n]) # extract n-gram from y_hat 
            if pred_gram not in grams:
                Cs.append(min(C(predicted, pred_gram), C(target, pred_gram))) # calculate C of each gram
                grams.add(pred_gram) # check if gram is already there
            
        
        numerator = np.sum(Cs) # TODO numerator of clipped precision
        denominator = len(predicted) - n + 1 # TODO denominator of clipped precision

        geo_mean *= (numerator/denominator)**(1/N)
    
    brevity_penalty = min(1, np.exp(1-len(target)/len(predicted))) # TODO
    return brevity_penalty * geo_mean


