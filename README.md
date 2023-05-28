# Wikipedia Language Classification 
#### I  will be investigating the use of decision trees and boosted decision stumps to classifytext as one of two languages. 
Specifically, my task is to collect data and train (in several different ways) somedecision stumps/trees so that when given a 15 word segment 
of text from either the English or Dutch Wikipedia, your code will state to the best of its ability which language the text is in.

## Data Collection 

The first step I will take is to collect some data. I can utilize the "Willekeurige pagina" (Random article) feature in the Dutch Wikipedia to gather information. 
To ensure the training set is suitable for testing, each example should consist of a 15-word sample. This means I can obtain multiple samples from a single page, 
but I should also gather data from different pages to represent various authors writing in Dutch. 

Now, I need to decide on the features to use for learning. Instead of feeding in 15 words of text as input, I will use features derived from those 15 words. 
Since decision trees require boolean (or few-valued) features, I will formulate boolean questions or attributes based on the text. For instance, 
I can consider features like "Does it contain the letter Q?" or "Is the average word length greater than 5?" 
It is important to come up with at least five distinct features.

Furthermore, I need to create the same features for the training data, test data, and novel test examples that will be provided during grading. 
I cannot count the same numeric feature as ten different attributes, but I can create multiple binary attributes from a single numeric feature if needed. 
If I'm unsure where to begin, I can start by identifying a few function words from each language and using them as potential features.


## Experimentation 

I will now proceed to implement the decision tree and Adaboost algorithms as specified. Here's an outline of the steps I will follow:

- Data Collection: I will gather the necessary data, including both training and test sets, to evaluate the performance of the algorithms.

- Decision Tree Algorithm: I will write code to create a decision tree based on the information gain algorithm discussed in the class and the book. 
The decision tree algorithm will accept weighted examples to accommodate Adaboost.

- Adaboost with Decision Trees: I will implement Adaboost using decision stumps. Adaboost relies on the underlying learning algorithm, 
so I will ensure that the decision tree algorithm accepts weighted examples.

- Evaluation: I will set aside a separate test set to assess the performance of both the decision tree and Adaboost algorithms. 
I will vary learning parameters such as depths, entropy cutoffs for the decision tree, and the number of stumps used in boosting. By measuring the error rate, I will determine the effectiveness of the algorithms given different features, training sizes, and learning parameters.

Throughout the implementation process, I will keep in mind the requirement to create the decision tree based on the information gain algorithm 
and to handle weighted examples for Adaboost.


## writeup.pdf in the directory gives the low level details of how I made sure to take care of these steps.
