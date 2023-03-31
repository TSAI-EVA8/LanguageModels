
import numpy as np
# =============================================================================
# Train
# =============================================================================
def get_batch(loader, loader_iter):
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)
    return batch, loader_iter

def training(data_loader,model,loss_model,optimizer,n_iteration):
    print('training...')
    print_each = 10
    model.train()
    batch_iter = iter(data_loader)
    #n_iteration = 1000
    losses=[]
    for it in range(n_iteration):
        
        #get batch
        batch, batch_iter = get_batch(data_loader, batch_iter)
        
        #infer
        masked_input = batch['input']
        masked_target = batch['target']
        
        masked_input = masked_input.cuda(non_blocking=True)
        masked_target = masked_target.cuda(non_blocking=True)
        output = model(masked_input)
        
        #compute the cross entropy loss 
        output_v = output.view(-1,output.shape[-1])
        target_v = masked_target.view(-1,1).squeeze()
        loss = loss_model(output_v, target_v)
        
        #compute gradients
        loss.backward()
        
        #apply gradients
        optimizer.step()
        
        losses.append(np.round(loss.item(),2))
        #print step
        if it % print_each == 0:
            print('it:', it, 
                ' | loss', np.round(loss.item(),2),
                ' | Δw:', round(model.embeddings.weight.grad.abs().sum().item(),3))
        
        #reset gradients
        optimizer.zero_grad()
        
    print("end of training")
    return losses


def eval(model,data_loader):
    model.eval()

    batch=next(iter(data_loader))

    #infer
    masked_input = batch['input']
    masked_target = batch['target']

    masked_input = masked_input.cuda(non_blocking=True)
    masked_target = masked_target.cuda(non_blocking=True)

    output = model(masked_input)
    
    return masked_input,masked_target,output