# How to use this RAG AI Teaching assistant an your own data 

# Step-1  collect your videos
Move all your video files to the videos folder

# Step-2 convert to mp3
Convert all the video files to mp3 by running video_to_mp3

# Step-3 Convert mp3 to json
Convert all the Mp3 files to json by running mp3_to_json 

# Step-4 Convert the json files to vectors 
Use the file preprocessing_json to convert json files to a dataframe with Embeddings and save it as a jobline pickle

# Step-5 Prompt generation anf feed to LLM
Read the Joblib file and load it into the memory. Then create a relavent prompt as per the user query and feed it to the LLm