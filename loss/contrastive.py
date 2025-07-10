import torch.nn as nn
import torch.nn.functional as F
import torch

class ContrastiveLoss(nn.Module):
    """
    Standard Contrastive Loss for image-text pairs
    Based on InfoNCE loss with temperature scaling
    
    Args:
        temperature (float): Temperature parameter to control the sharpness of the distribution
        use_hard_negatives (bool): Whether to use hard negative mining
        margin (float): Margin for hard negative mining (if enabled)
    """
    def __init__(self, temperature=0.07, use_hard_negatives=False, margin=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.use_hard_negatives = use_hard_negatives
        self.margin = margin

    def forward(self, image_features, text_features, batch_size=None):
        # Normalize features
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        # Compute cosine similarity between image and text features
        logits_per_image = torch.matmul(image_features, text_features.t()) / self.temperature
        logits_per_text = logits_per_image.t()

        # Create mask for positive pairs (diagonal)
        batch_size = logits_per_image.size(0)
        mask = torch.eye(batch_size, dtype=torch.bool, device=image_features.device)

        if self.use_hard_negatives:
            return self._hard_negative_loss(logits_per_image, logits_per_text, mask)
        else:
            return self._standard_contrastive_loss(logits_per_image, logits_per_text, mask)

    def _standard_contrastive_loss(self, logits_per_image, logits_per_text, mask):
        """
        Standard InfoNCE contrastive loss
        """
        # Image-to-text loss
        exp_logits = torch.exp(logits_per_image)
        log_prob = logits_per_image - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_img = (mask * log_prob).sum(1) / mask.sum(1)

        # Text-to-image loss
        exp_logits = torch.exp(logits_per_text)
        log_prob = logits_per_text - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_text = (mask * log_prob).sum(1) / mask.sum(1)

        # Total loss
        loss = -(mean_log_prob_img + mean_log_prob_text) / 2
        return loss.mean()

    def _hard_negative_loss(self, logits_per_image, logits_per_text, mask):
        """
        Contrastive loss with hard negative mining
        """
        # Find hard negatives (closest negative pairs)
        neg_mask = ~mask
        
        # Image-to-text hard negatives
        neg_logits_img = logits_per_image * neg_mask.float()
        hard_neg_img = torch.topk(neg_logits_img, k=min(3, neg_mask.sum(1).min().item()), dim=1)[0]
        
        # Text-to-image hard negatives
        neg_logits_text = logits_per_text * neg_mask.float()
        hard_neg_text = torch.topk(neg_logits_text, k=min(3, neg_mask.sum(1).min().item()), dim=1)[0]

        # Positive pairs
        pos_img = (logits_per_image * mask.float()).sum(1)
        pos_text = (logits_per_text * mask.float()).sum(1)

        # Contrastive loss with hard negatives
        loss_img = -torch.log(torch.exp(pos_img) / (torch.exp(pos_img) + torch.exp(hard_neg_img + self.margin).sum(1)))
        loss_text = -torch.log(torch.exp(pos_text) / (torch.exp(pos_text) + torch.exp(hard_neg_text + self.margin).sum(1)))

        return (loss_img + loss_text) / 2


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent)
    A variant of contrastive loss commonly used in self-supervised learning
    """
    def __init__(self, temperature=0.5, normalize=True):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.normalize = normalize

    def forward(self, image_features, text_features):
        if self.normalize:
            image_features = F.normalize(image_features, p=2, dim=1)
            text_features = F.normalize(text_features, p=2, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(image_features, text_features.t()) / self.temperature

        # Create labels (diagonal elements are positive pairs)
        batch_size = similarity_matrix.size(0)
        labels = torch.arange(batch_size, device=image_features.device)

        # Compute loss for both directions
        loss_i2t = F.cross_entropy(similarity_matrix, labels)
        loss_t2i = F.cross_entropy(similarity_matrix.t(), labels)

        return (loss_i2t + loss_t2i) / 2


class TripletLoss(nn.Module):
    """
    Triplet Loss for contrastive learning
    """
    def __init__(self, margin=0.3, distance='cosine'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance = distance

    def forward(self, image_features, text_features, labels=None):
        if self.distance == 'cosine':
            # Normalize features for cosine distance
            image_features = F.normalize(image_features, p=2, dim=1)
            text_features = F.normalize(text_features, p=2, dim=1)
            
            # Compute cosine similarity
            similarity = torch.matmul(image_features, text_features.t())
            distance = 1 - similarity
        else:
            # Euclidean distance
            distance = torch.cdist(image_features, text_features, p=2)

        # If no labels provided, assume diagonal pairs are positive
        if labels is None:
            batch_size = image_features.size(0)
            labels = torch.arange(batch_size, device=image_features.device)

        # Find positive and negative pairs
        pos_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        neg_mask = ~pos_mask

        # Get positive distances
        pos_dist = distance[pos_mask]

        # Get hardest negative distances
        neg_dist = distance[neg_mask].view(distance.size(0), -1)
        hardest_neg_dist = neg_dist.min(dim=1)[0]

        # Triplet loss
        loss = torch.clamp(pos_dist - hardest_neg_dist + self.margin, min=0.0)
        
        return loss.mean() 