def get_default_config(data_name):
    if data_name in ['Scene-15']:
        return dict(
            Autoencoder=dict(
                arch1=[59, 1024, 1024, 1024, 16],
                arch2=[20, 1024, 1024, 1024, 16],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
                d_model=16,
                nhead=2,
                dim_feedforward=128,
                num_layers=2,
            ),
            training=dict(
                missing_rate=0.3,
                seed=8,
                batch_size=512,
                epoch=500,
                lr=1.0e-4,
                lambda1=10,
                lambda2=0.1,
                lambda3=0.1,
                lambda4=0.1,
                lambda5=0.01,
                lambda6=0.5,
                kernel_mul=0.01,
                kernel_num=3,
            ),
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            pretraining=dict(
                epochs=100,  # 预训练的轮数
                batch_size=1024,  # 预训练的批量大小
              # 跨视图预测损失的权重
            )
        )
    else:
        raise Exception('Undefined data_name')
