import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', size=15) 

def visualize_tensor(tensor, title=""):
    tensor = tensor.cpu().numpy().transpose(1, 2, 0)
    plt.imshow(tensor.astype('uint8'))
   # plt.title(title)
    plt.axis('off')
    plt.show()


class PadPrompter(nn.Module):
    def __init__(self, args):
        super(PadPrompter, self).__init__()
        self.pad_size = args.prompt_size
        image_size_w = 640
        image_size_h = 483

        self.base_size_w = image_size_w - self.pad_size*2
        self.base_size_h = image_size_h - self.pad_size*2 
        self.pad_up = nn.Parameter(torch.randn([1, 3, self.pad_size, image_size_w]))
        self.pad_down = nn.Parameter(torch.randn([1, 3, self.pad_size, image_size_w]))  
        self.pad_left = nn.Parameter(torch.randn([1, 3, self.base_size_h, self.pad_size]))
        self.pad_right = nn.Parameter(torch.randn([1, 3, self.base_size_h, self.pad_size]))

    def forward(self, x):
        base = torch.zeros(1, 3, self.base_size_h, self.base_size_w).cuda()
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)

        zero_padding = torch.zeros(1, 3, 640 - self.base_size_h - 2*self.pad_size, self.base_size_w + 2*self.pad_size).cuda()
        prompt = torch.cat([prompt, zero_padding], dim=2)

        prompt = torch.cat(x.size(0) * [prompt])

        return x + prompt, prompt
    
        # visualize_tensor(x[0], title='x')
        
        # x_plus_prompt, prompt = x + prompt * 700, prompt * 700
        # x_plus_prompt = x_plus_prompt.cpu()
        # prompt = prompt.cpu()
        # prompt[prompt == 0] = 255
       
        # ##x + prompt 시각화 (클램핑을 사용하여 값 범위를 0~255로 조정)
        # x_plus_prompt_clamped = torch.clamp(x_plus_prompt, 0, 255)
        # visualize_tensor(x_plus_prompt_clamped[0], title="x + prompt")

        # # prompt 시각화 (0~255 범위로 리스케일링)
        # prompt_rescaled = torch.clamp(prompt, 0, 255)
        # visualize_tensor(prompt_rescaled[0], title="Prompt")

        # x_plus_prompt = x_plus_prompt.cuda()
        # prompt = prompt.cuda()
        
        # return x_plus_prompt, prompt

class FixedPatchPrompter(nn.Module):
    def __init__(self, args):
        super(FixedPatchPrompter, self).__init__()
        self.isize = args.image_size
        self.psize = args.prompt_size
        self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))

    def forward(self, x):
        prompt = torch.zeros([1, 3, self.isize, self.isize]).cuda()
        prompt[:, :, :self.psize, :self.psize] = self.patch

        return x + prompt, prompt


class RandomPatchPrompter(nn.Module):
    def __init__(self, args):
        super(RandomPatchPrompter, self).__init__()
        self.isize = args.image_size
        self.psize = args.prompt_size
        self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))

    def forward(self, x):
        x_ = np.random.choice(self.isize - self.psize)
        y_ = np.random.choice(self.isize - self.psize)

        prompt = torch.zeros([1, 3, self.isize, self.isize]).cuda()
        prompt[:, :, x_:x_ + self.psize, y_:y_ + self.psize] = self.patch

        return x + prompt, prompt


def padding(args):
    return PadPrompter(args)


def fixed_patch(args):
    return FixedPatchPrompter(args)


def random_patch(args):
    return RandomPatchPrompter(args)
