import streamlit as st
from PIL import Image
import torch
import pickle
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import transforms

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        
        self.freq_threshold = freq_threshold
    
    def __len__(self):
        return len(self.itos)
    
    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]
    
    def build_vocabulary(self,sentences):
        idx = 4
        frequency = {}
        
        for sentence in sentences:
            for word in self.tokenizer_eng(sentence):
                if word not in frequency:
                    frequency[word] = 1
                else:
                    frequency[word] += 1
                
                if (word not in self.stoi and frequency[word] > self.freq_threshold-1):
                    self.itos[idx] = word
                    self.stoi[word] = idx
                    idx += 1
    
    def numericalize(self,sentence):
        tokenized_text = self.tokenizer_eng(sentence)
        
        return [self.stoi[word] if word in self.stoi else self.stoi["<UNK>"] for word in tokenized_text ]
                    

class EncoderCnn(nn.Module):
    def __init__(self,embed_size):
        super(EncoderCnn, self).__init__()
        
        #using pretrained model 
        pretrained = models.resnet50(weights=True)
        #freezing training of weights
        for param in pretrained.parameters():
            param.requires_grad = False
            
        #getting all layers in list module and removed the last layer(we will add our own)
        modules = list(pretrained.children())[:-1]
        #putting final layers in out variable
        self.pretrained = nn.Sequential(*modules)
        
        #changing last layer and allow it to train
        self.fc = nn.Linear(pretrained.fc.in_features, embed_size)
        self.batch= nn.BatchNorm1d(embed_size,momentum = 0.01)
        self.fc.weight.data.normal_(0., 0.02)
        self.fc.bias.data.fill_(0)
        
    def forward(self,images):
        features = self.pretrained(images)
        #flattening to pass to fully connected layer
        features = features.view(features.size(0),-1)
        features = self.batch(self.fc(features))
        return features  #[batch, embed_size]
    

 
class DecoderRnn(nn.Module):
    def __init__(self, embed_size,vocab_size, hidden_size, num_layers):
        super(DecoderRnn, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size) #[batch,seq] -> [batch,seq,embed]
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,batch_first=True) #[batch,seq,embed]->hiddens[batch,seq,hidden_size]
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, features, caption):
        embeddings = self.dropout(self.embedding(caption)) #[batch,seq] -> [batch,seq,embed]
        embeddings = torch.cat((features.unsqueeze(1),embeddings), dim=1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

 
class CNN2RNN(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size, num_layers):
        super(CNN2RNN, self).__init__()
        self.encoderCNN = EncoderCnn(embed_size)
        self.decoderRNN = DecoderRnn(embed_size, vocab_size, hidden_size, num_layers)
    
    def forward(self, images, caption):
        x = self.encoderCNN(images)
        x = self.decoderRNN(x, caption)
        return x
    
    def captionImage(self, image, vocabulary, maxlength=50):
        result_caption = []
        
        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            print(x)
            states = None
            
            for _ in range(maxlength):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                result_caption.append(predicted.item())
                x = self.decoderRNN.embedding(predicted).unsqueeze(0)
                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break
        return [vocabulary.itos[i] for i in result_caption]

 
embed_size = 256
hidden_size = 256
num_layers = 2

 
# Load the image captioning model
def load_model(model_path,vocab):
    model = CNN2RNN(embed_size=embed_size, hidden_size=hidden_size,vocab_size=len(vocab.stoi), num_layers=num_layers)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

# Load vocabulary from pickle file
def load_vocabulary(vocabulary_path):
    with open(vocabulary_path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab

# Define image preprocessing transformation
transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

def generate_caption(image, model, vocab):
    # Preprocess the image
    image = transform(image).unsqueeze(0)

    # Generate caption
    caption = model.captionImage(image, vocab)

    return caption

def main():
    st.title("Image Captioning App")

    # Upload image
    uploaded_image = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Load model and vocabulary
        model_path = 'model.pth.tar'  # Path to your model checkpoint
        vocabulary_path = "vocabulary.pkl"  # Path to your vocabulary pickle file
        vocab = load_vocabulary(vocabulary_path)
        model = load_model(model_path,vocab)
        

        # Generate caption button
        if st.button('Generate Caption'):
            # Generate caption for the uploaded image
            cap = generate_caption(image, model, vocab)
            caption = ""
            for i in cap:
                if i!= '<SOS>' and i!='<EOS>':
                    caption+=i
                    caption+=' '
            st.write('**Generated Caption:**', caption)

if __name__ == '__main__':
    main()



# command to run in local : streamlit run app.py --server.enableXsrfProtection false