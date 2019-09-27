#### <a href="https://cs.stanford.edu/people/karpathy/main.pdf" target="_blank">《Connecting Images And Natural Language》</a>Andrej Karpathy (2016)
How can we endow computers with an understanding of so many interconnected abstract concepts and knowledge?  

First, it is important to recognize that a necessary (but not sucient) condition is that the information about the world must be made available to the computer. This already presents many practical diculties related to data collection and storage. For instance, in this concrete example, I recognize Obama from news articles and TV, and a computer cannot do so until it somehow gains access to the same data. As a more problematic example, I understand how the mechanical scale in image works because I’ve interacted with one myself in the real world: I stood on it and saw its reading change, I played with the setting of its counter weights, I shifted my weight around and saw it react. Therefore, an argument can be made that I’ve benefited a great deal from embodied interaction with the environment and my ability to run small experiments that disambiguate between di↵erent likely hypotheses of world dynamics. This line of thought argues that we might not reach the same level of understanding in computers until they can also interact with the world in the same way we have done for many years of our upbringing.  

Second, it is insightful to note that the representations of these abstract relationships are dicult to manually encode in some formal language and provide to the computer directly under the umbrella of supervised learning. For instance, CYC [59] is a popular example of an attempt to assemble a comprehensive ontology and knowledge base of common knowledge in a formal language, which turned out to be very challenging. Instead, it seems that a more promising approach is to allow a model to discover its own internal representations, similar to word embedding methods where the structure and relationships emerge as a result of optimizing some objective.  

Lastly, one central challenge lies in how we can design architectures that can model abstract concepts and theories (e.g. that people have a finite field of view, or that scales measure weight) and how they can be acquired, stored and manipulated in an end-to-end learning framework. Furthermore, one would like to represent distributions over theories and come up with objectives that encourage agents to disambiguate between competing hypotheses (e.g. figuring out how a scale works).   

To make further progress, it is these kinds of problems that we must turn to next.