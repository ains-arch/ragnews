# pylint: disable=W1514
# I don't know what encoding to specify for these files and I'm too lazy to check
# pylint: disable=C0103
# They're not constants, pylint

'''
This file is for evaluating the quality of our RAG system using the Hairy Trumpet tool/dataset
'''

import json
import argparse
import re
import ragnews

def pull_labels(path):
    """
    This function extracts all possible labels from the Hairy Trumpet data file.
    
    Arguments:
        path (str): The path to the Hairy Trumpet data file.
    
    Returns:
        labels (set): A set of unique labels from the dataset.
    """
    labels = set()
    with open(path) as fin:
        for inner_line in fin:
            dp = json.loads(inner_line)
            labels.update(dp['masks'])
    return labels

class RAGClassifier: # pylint: disable=too-few-public-methods
    """
    Predictor class following the scikit-learn interface. Uses the ragnews.rag function to perform
    the prediction.
    
    Arguments:
        labels (set): Specifies the valid labels to predict.
    """
    def __init__(self, labels):
        self.labels = " ".join(labels)
        self.db = ragnews.ArticleDB('ragnews/ragnews.db')

    def predict(self, class_masked_text):
        """
        >>> model = RAGClassifier()
        >>> model.predict('There is no mask token here.')
        []
        >>> model.predict('[MASK0] is the democratic nominee')
        ['Harris']
        >>> model.predict('[MASK0] is the democratic nominee and [MASK1] is the republican nominee')
        ['Harris', 'Trump']
        
        Arguments:
            self (RAGClassifier object)
            class_masked_text (str): The input string that includes MASK tokens to refill.
        
        Returns:
            Call to ragnews.rag, which is hopefully a list of string(s) for best token choice(s) to
            refill the MASK(s).
        """

        textprompt = f'''
This is a fancier question that is based on standard cloze style benchmarks.
I'm going to provide you a sentence, and that sentence will have a masked token inside of it that
will look like [MASK0].
And your job is to tell me what the value of that masked token was.
Valid values include: {self.labels}

The size of your output should just be a single word for the mask.
You should not provide any explanation or other extraneous words.
If there are no news articles given, just make an educated guess based on the valid values.

INPUT: [MASK0] is the democratic nominee.
OUTPUT: Harris

INPUT: {class_masked_text}
OUTPUT: '''

        return ragnews.rag(textprompt, self.db, keywords_text=class_masked_text)

# When the file is run as a script
if __name__ == '__main__':
    # Take a command line argument which is the path to the hairytrumpet data file
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', required=True, help='Path to the Hairy Trumpet dataset file')
    args = parser.parse_args()

    # Extract all possible labels from the dataset
    name_labels = pull_labels(args.path)
    # print(f"DEBUG: name_labels: {name_labels}")

    # Initialize the classifier with the extracted labels
    classifier = RAGClassifier(name_labels)

    # Evaluate the classifier on the dataset
    correct = 0
    total = 0
    with open(args.path) as json_file:
        for line in json_file:
            current_line = json.loads(line)
            # print(f"DEBUG: current_line: {current_line}")
            masked_text = current_line.get('masked_text', '')
            # print(f"DEBUG: masked_text: {masked_text}")
            true_labels = current_line.get('masks', [])
            # print(f"DEBUG: true_labels: {true_labels}")

            if not masked_text or not true_labels:
                continue

            # Run the predict method on each instance inside the file
            pred = classifier.predict(masked_text)
            # print(f"DEBUG: raw pred = {pred}")

            # Remove unwanted characters like square brackets or quotes
            cleaned_pred = re.sub(r"[\[\]']", '', pred)
            # print(f"DEBUG: cleaned_pred: {cleaned_pred}")

            # Split the predictions by common delimiters (comma, space, or newline)
            predictions = re.split(r'[,\n\s]+', cleaned_pred.strip())
            # print(f"DEBUG: predictions: {predictions}")

            # Filter out empty predictions
            relevant_predictions = [p.strip() for p in predictions if p.strip()]
            # print(f"DEBUG: relevant_predictions: {relevant_predictions}")

            # Compare predictions to true labels
            correct += sum(1 for pred, true in zip(relevant_predictions, true_labels) if pred ==
                           true)
            # print(f"DEBUG: correct: {correct}")
            total += len(true_labels)
            # print(f"DEBUG: total: {total}")
            intermediate_accuracy = correct / total if total > 0 else 0
            # print(f"DEBUG: intermediate_accuracy: {intermediate_accuracy}", flush=True)

    accuracy = correct / total if total > 0 else 0
    print(accuracy)
