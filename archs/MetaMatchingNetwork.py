import torch
import torch.nn.functional as F
from utils.Knowledge_Distillation_Loss import KnowledgeDistillationLoss

# Meta Matching Network
class MetaMatchingNetwork():
    def __init__(self, cfg):
        self.cfg = cfg

    def update_meta_weights_only(self, MetaN, meta_optimizer, grad_meta):
        for weight, grad_item in zip(MetaN.parameters(), grad_meta):
            weight.grad = grad_item

        # clip gradients
        torch.nn.utils.clip_grad_norm_(MetaN.parameters(), 1)

        meta_optimizer.step()
        for weight, grad_item in zip(MetaN.parameters(), grad_meta):
            del weight.grad

    # simulate sgd updating
    def simulate_sgd_update(self, w, g, optimizer):
        return -g * optimizer.param_groups[-1]['lr'] + w


    def calculate_kd_gradient(self, kd_loss, selected_params, gen_kd_optimizer):
        gen_kd_optimizer.zero_grad()
        grad = torch.autograd.grad(
            kd_loss,
            selected_params,
            create_graph=True,
            allow_unused=True
        )
        return grad

    def calculate_meta_gradient(self, g_loss_adv, gen_net, MetaN, meta_optimizer, random_cand, students_weight):
        meta_optimizer.zero_grad()
        grad_student_val = torch.autograd.grad(
            g_loss_adv, gen_net.module.get_subnet_parameters(random_cand), retain_graph=True, allow_unused=True)

        grad_meta = torch.autograd.grad(
            students_weight[0],
            MetaN.parameters(),
            grad_outputs=grad_student_val,
            allow_unused=True)
        return grad_meta

    # forward training data
    def forward_training(self, x, gen_net, random_cand, teacher_cand, meta_value):
        student_output, student_intermediates = gen_net(x, random_cand)
        with torch.no_grad():
            teacher_output, teacher_intermediates = gen_net(x, teacher_cand)
            teacher_output.detach()
            soft_label = F.softmax(teacher_output, dim=1)
        kd_loss = meta_value * KnowledgeDistillationLoss(student_output, soft_label)
        return kd_loss

    def isUpdate(self, current_epoch, batch_idx, prioritized_board):
        isUpdate = True
        isUpdate &= (current_epoch > self.cfg.META_STA_EPOCH)
        isUpdate &= (batch_idx > 0)
        isUpdate &= (batch_idx % self.cfg.PRIORITIZED_BOARD_UPDATE_ITER == 0)
        isUpdate &= (prioritized_board.board_size() > 0)
        return isUpdate

    # update meta matching networks
    def run_update(self, gen_z, random_cand, gen_net, gen_kd_optimizer, dis_net, MetaN, meta_optimizer,
                   prioritized_board, current_epoch, batch_idx):
        if self.isUpdate(current_epoch, batch_idx, prioritized_board):
            # x = self.get_minibatch_input(input)
            gen_kd_optimizer.zero_grad()

            meta_value, teacher_cand = prioritized_board.select_teacher(gen_net, MetaN, random_cand)
            kd_loss = self.forward_training(gen_z, gen_net, random_cand, teacher_cand, meta_value)
            kd_loss.backward()

            gen_kd_optimizer.step()

            gen_imgs, _ = gen_net(gen_z, random_cand)
            fake_validity = dis_net(gen_imgs)
            g_loss_adv = -torch.mean(fake_validity)

            # calculate meta gradient
            students_weight = gen_net.module.get_subnet_parameters(random_cand)
            grad_meta = self.calculate_meta_gradient(g_loss_adv, gen_net, MetaN, meta_optimizer, random_cand, students_weight)

            # update meta matching networks
            self.update_meta_weights_only(MetaN, meta_optimizer, grad_meta)

            # delete internal variants
            del grad_meta, gen_z, g_loss_adv, kd_loss, students_weight