# RelRepair: :  Retrieving Relevant Information to Enhance Automated Program Repair
RelRepair is a novel Retrieval-Augmented Generation (RAG) framework that improves automated program repair by retrieving relevant information and then steering an LLM to generate higher-quality patches.
## Prerequisites
<pre> <code>
  Install Defects4J from https://github.com/rjust/defects4j 
  export PATH=$PATH:"path2defects4j"/framework/bin 
</code> </pre>

<pre> <code>
sudo apt-get install openjdk-8-jdk -y
export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
export CLASSPATH=.:$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar
</code> </pre>

<pre> <code>
numpy==1.24.3
pandas==2.0.3
torch==2.0.1
torchvision==0.15.2
transformers==4.29.2
openai==1.30.1
sentence_transformers==2.6.1
scikit-learn==1.4.2
</code> </pre>
