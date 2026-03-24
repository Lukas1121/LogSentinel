import torch, json

ckpt = torch.load('data/tenant_test/finetuned/model_finetuned.pt', map_location='cpu', weights_only=True)
print('Checkpoint epoch:   ', ckpt.get('epoch'))
print('Checkpoint val_loss:', ckpt.get('val_loss'))

log = json.load(open('data/tenant_test/finetuned/finetune_log.json'))
print(f'\nTotal epochs logged: {len(log)}')
print('Last 5 epochs:')
for e in log[-5:]:
    ep    = e['epoch']
    phase = e['phase']
    vl    = e['val_loss']
    print(f'  ep{ep} {phase}  val_loss={vl:.4f}')
