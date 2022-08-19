
class ReconstructedLoss(object):
    def __init__(self, loss_opt, loss_adj, w_opt=1.0, w_adj=1.0):
        super(ReconstructedLoss).__init__()
        self.loss_opt = loss_opt
        self.loss_adj = loss_adj
        self.w_opt = w_opt
        self.w_adj = w_adj

    def __call__(self, inputs, target):
        opt_recon, adj_recon = inputs
        opt, adj = target
        loss_opt = self.loss_opt(opt_recon, opt)
        loss_adj = self.loss_adj(adj_recon, adj)
        loss = self.w_opt * loss_opt + self.w_adj * loss_adj
        return loss
