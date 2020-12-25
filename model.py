import torch.nn as nn


""" Optional conv block """
def conv_block(in_channels, out_channels, kernel_size):

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(
                kernel_size=kernel_size,
                stride=kernel_size,
                ceil_mode=False,
            ),
    )


""" Define your own model """
class FewShotModel(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super(FewShotModel,self).__init__()
        self.conv1 = conv_block(x_dim, hid_dim, 2) # N x 64 x 112 x 112
        self.conv2 = conv_block(hid_dim, hid_dim, 2) # N x 64 x 56 x 56
        self.conv3 = conv_block(hid_dim, hid_dim, 2) # N x 64 x 28 x 28
        self.conv4 = conv_block(hid_dim, hid_dim, 2) # N x 64 x 14 x 14
        self.fc1 = nn.Linear(64*14*14, 1024)
        #self.fc2 = nn.Linear(1024, 256)
        #self.fc3 = nn.Linear(256, z_dim)
        #self.fc4 = nn.Linear(256, z_dim)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        size = x.size()[1:]
        n_features = 1
        for s in size:
            n_features *= s
        x = x.view(-1, n_features)
        
        x = self.fc1(x)
        #x = self.fc2(x)
        #x = self.fc3(x)
        #x = self.fc4(x)         
                   
        return x
    
   

""" Below code was not used """
""" Define your own model """
class CNNEncoder(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super(FewShotModel,self).__init__()
        self.conv1 = conv_block(x_dim, hid_dim, 2) # N x 64 x 112 x 112
        self.conv2 = conv_block(hid_dim, hid_dim, 2) # N x 64 x 56 x 56
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)                   
        return x
    
class RelationNetwork(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super(FewShotModel,self).__init__()
        self.conv1 = conv_block(hid_dim*2, hid_dim, 2) # N x 64 x 28 x 28
        self.conv2 = conv_block(hid_dim, hid_dim, 2) # N x 64 x 14 x 14
        self.fc1 = nn.Linear(64*14*14, 1024)
        self.fc2 = nn.Linear(64*14*14, 1024)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        size = x.size()[1:]
        n_features = 1
        for s in size:
            n_features *= s
        x = x.view(-1, n_features)
        
        x = self.fc1(x)
        return x
    
    