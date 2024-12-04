import torch
import torch.nn as nn
import torch.nn.functional as F

class DKVMN(nn.Module):
    def __init__(self, memory_size, key_dim, value_dim, question_embed_dim, value_embed_dim):
        super(DKVMN, self).__init__()
        self.memory_size = memory_size
        self.key_dim = key_dim
        self.value_dim = value_dim
        
        # Embeddings
        self.question_embed = nn.Embedding(100, question_embed_dim)  # 100 is a placeholder for the question vocab size
        self.value_embed = nn.Embedding(100, value_embed_dim)  # 100 is a placeholder for the value vocab size
        
        # Memory and attention
        self.attention = nn.Linear(question_embed_dim, memory_size)  # For attention computation
        self.fc = nn.Linear(question_embed_dim + value_dim, 1)  # Output prediction

    def forward(self, questions, answers):
        seq_len = questions.size(0)  # Length of the sequence
        batch_size = questions.size(1)  # Batch size
        
        # Initialize memory
        batch_memory_value = torch.zeros(batch_size, self.memory_size, self.value_dim).to(questions.device)  # (batch_size, memory_size, value_dim)
        
        predictions = []
        
        for t in range(seq_len):
            q = questions[t]  # (batch_size,)
            a = answers[t]    # (batch_size,)
            
            # Embed question and answer
            q_embed = self.question_embed(q)  # (batch_size, key_dim)
            attention_weights = F.softmax(self.attention(q_embed), dim=-1)  # (batch_size, memory_size)
            
            # Read from memory
            read_content = torch.bmm(attention_weights.unsqueeze(1), batch_memory_value).squeeze(1)  # (batch_size, value_dim)
            
            # Combine with current question embedding
            combined_content = torch.cat([q_embed, read_content], dim=1)  # (batch_size, key_dim + value_dim)
            
            # Predict the answer for the current question
            pred = self.fc(combined_content).squeeze(-1)  # (batch_size,)
            predictions.append(pred)
            
            # Embed question-answer pair for memory update
            # Ensure the indices are within range
            index = (q * 2 + a) % 100  # This will ensure the index stays within [0, 99]
            qa_embed = self.value_embed(index)  # (batch_size, value_dim)

            # Write to memory (update memory values)
            batch_memory_value = batch_memory_value + torch.bmm(attention_weights.unsqueeze(2), qa_embed.unsqueeze(1)).squeeze(1)
        
        return torch.stack(predictions, dim=0)  # (seq_len, batch_size)

# Example test
questions = torch.randint(0, 100, (5, 32))  # 5 questions, batch size 32
answers = torch.randint(0, 100, (5, 32))  # 5 answers, batch size 32

model = DKVMN(memory_size=10, key_dim=50, value_dim=50, question_embed_dim=50, value_embed_dim=50)
predictions = model(questions, answers)
print(predictions.shape)  # Expected output: (5, 32)