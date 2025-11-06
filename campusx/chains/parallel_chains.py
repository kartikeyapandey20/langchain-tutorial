from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

llm1 = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation"
)

llm2 = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation"
)

model1 = ChatHuggingFace(llm=llm1)

model2 = ChatHuggingFace(llm=llm2)


prompt1 = PromptTemplate(
    template="Generate short and simple notes form the following text \n {text}",
    input_variables=['text'])


prompt2 = PromptTemplate(
    template="Genrate short question and answer from the following text. \n {text}",
    input_variables=['text'])

prompt3 = PromptTemplate(
    template="Merge the provide notes ans quiz into a single document. \n notes-> {notes} and quiz -> {quiz}",
    input_variables=['text'])


parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes' : prompt1 | model1 | parser,
    'quiz' : prompt2 | model2 | parser
})


merge_chain = prompt3 | model2 | parser

chain = parallel_chain | merge_chain

text = """Support vector machines (SVMs) are a set of supervised learning methods used for classification,
regression and outliers detection.

The advantages of support vector machines are:

· Effective in high dimensional spaces.
· Still effective in cases where number of dimensions is greater than the number of samples.
. Uses a subset of training points in the decision function (called support vectors), so it is also memory
efficient.

· Versatile: different Kernel functions can be specified for the decision function. Common kernels are
provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

. If the number of features is much greater than the number of samples, avoid over-fitting in choosing
Kernel functions and regularization term is crucial.
· SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold
cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by
numpy.asarray ) and sparse (any scipy. sparse) sample vectors as input. However, to use an SVM to make
predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered
numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64"""

result = chain.invoke({
    'text' : text
})

print(result)