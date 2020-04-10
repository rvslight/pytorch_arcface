from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features)) # in_features=512,, out_features.. num of classes
        nn.init.xavier_uniform_(self.weight) ## 즉.. 각 feature별로 고르게 분포.. 각 클래스별로.. 고른 분포로.. 중심점을 갖게 해줄 수 있다.. feature공간에서..

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        ### Ground truth target에 각도 m만큼 패널티를 부과해서.. 90도에서 270도 바깥에 있게 하겠다.. 기준은 W로 부터.. .. W는 각 class의 하나의 중심점 역할을 한다.
        ## 각 클래스별 고른 공간에.. 중심이.. 각각.. 분포되어 있고.. 결국에는 각 클래스별 중심점에서 멀리 90도에서 270도 사이에 모이게 하는 역할을 준다.
        ## 즉.. 마진을 더하서면서까지.. 정답을 맞추게 혹독하게 만든다. 즉.. 각 균일한 클래스 중심점에서 일정범위에서 모이게 하는 조건을 유지하면서도 정답을 맞추게.. 즉.. 마진이 생긴다.
        cosine = F.linear(F.normalize(input), F.normalize(self.weight)) ## cosine => fc7
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1)) ## origianl_target_logit
        phi = cosine * self.cos_m - sine * self.sin_m ## marginal_target_logit (코사인(알파+베타) = 코사인(알파)x코사인(베터) - 싸인(알파)x싸인(베타)
        ##즉, 원레 각도.. cosine(fc7)의 각도에 m을 더한 각도를 phi로 함..
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine) ##양이면.. 마진을 더하고,, 음이면 그냥 놔둔다.(즉.. 90도에서 270도는 방관?)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm) ##어떤 기준점.. 보다 크면,, 마진을 더하고, 작으면 각도를 줄인다..
            ## 음수여도 뭔가.. 조작을..
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1) # scatter( dim.. labeliew is channe.. horizontal.. , last element. is value).. so [B,C] one hot vector..
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        ## 정답의 위치를 계속해서.. marginal_target_logit으로 가게 만드는 역할...!! 장답을.. fc7 에 math.cos(m)을 곱한쪽으로 가게끔 한다..
        ## 1에서 원핫을 빼면.. 결과는.. 트루에는 phi.. 거짓.. 그곳에는 fc7..이 잔존.. 즉.. 투루에만 먼가 강조한다..
        ## cosine.. 는 fc7인데.. 그 곳에.. 즉, phi가.. marginal_target_logit - original_target_logit을 더한다..
        ## just add something.. in true class feature.. phi .. and cosine..
        ### 정답클래스를.. . 뭔가 강조하는 느낌.. / 오답클래스는 보존하고..
        # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        ## 곱하는 것은.. s를 곱하는 것은 값의 차를 좀 더 크게 준다는 의미.. 순위에는 변화는 없다.
        # print(output)

        return output


class AddMarginProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


class SphereProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        m: margin
        cos(m*theta)
    """
    def __init__(self, in_features, out_features, m=4):
        super(SphereProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform(self.weight)

        # duplication formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input, label):
        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        self.iter += 1
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.m * theta / 3.14159265).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(input, 2, 1)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cos_theta.size())
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # --------------------------- Calculate output ---------------------------
        output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
        output *= NormOfFeature.view(-1, 1)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) + ')'
