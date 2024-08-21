# import torch
#
# labels =[1, 0]
# logist = [ 1, 34]
#
#
#
# # logist = torch.softmax(logist, dim=-1)
# i = 0
# eps = 1e-6
#
# logist[i] = (torch.exp(logist[i]) + eps) / torch.sum(torch.exp(logist))
#
# boundary, _ = torch.max(logist)
# loss = (- labels * (torch.log(logist / boundary + eps) + torch.log(boundary))
#         - (1 - labels) * torch.log(torch.clamp((1-logist), eps, 1.0)))
# loss = torch.mean(loss)
#
#
#
# loss = (- labels * r * (1-logist) * (torch.log(logist)
#         - (1 - labels) * torch.log(torch.clamp((1-logist), eps, 1.0)))


a = [1, 2, 3, 4]
b = [1, 2, 4, 5]

dp = [[0 for j in range(len(b))] for i in range(len(a))]
#dp[i][j]

for i in range(len(a)):
    for j in range(len(b)):
        if i > j:
            if a[i] == b[j]:
                dp[i][j] = dp[i][j - 1] - 1
            else:
                dp[i][j] = dp[i][j - 1] + 1
        else:
            if a[i] == a[j]:
                dp[i][j] = dp[i-1][j] - 1
            else:
                dp[i][j] = dp[i - 1][j] + 1

print(dp[-1][-1])