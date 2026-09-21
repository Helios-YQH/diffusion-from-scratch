## Training cost

| Cell | Run | Backbone | Objective | Params (M) | GFLOPs/sample | s/step | img/s | Peak GB | MFU | Best loss |
|---|---|---|---|---|---|---|---|---|---|---|
| ? | `mnist_dit_eps` | dit | eps | 16.5 | 1.08 | 0.073 | 3526 | 2.8 | 7% | 0.0216 |
| ? | `mnist_dit_rf` | dit | rf | 16.5 | 1.08 | 0.072 | 3576 | 2.8 | 7% | 0.1669 |
| ? | `mnist_unet_eps` | unet | eps | 11.7 | 2.43 | 0.101 | 2545 | 3.5 | 12% | 0.0193 |
| ? | `mnist_unet_rf` | unet | rf | 11.7 | 2.43 | 0.102 | 2507 | 3.5 | 12% | 0.1568 |


## Generation quality (FID)

| Cell | Run | Sampler | NFE | n | Weights | FID |
|---|---|---|---|---|---|---|
| ? | `ddpm_celeba` | ddim | 10 | 10000 | ema | 31.398 |
| ? | `ddpm_celeba` | ddim | 20 | 10000 | ema | 19.133 |
| ? | `ddpm_celeba` | ddim | 50 | 10000 | ema | 12.748 |
| ? | `ddpm_celeba` | ancestral | 1000 | 10000 | ema | 11.367 |
| ? | `mnist_dit_eps` | ddim | 10 | 10000 | ema | 49.136 |
| ? | `mnist_dit_eps` | ddim | 20 | 10000 | ema | 159.679 |
| ? | `mnist_dit_eps` | ddim | 20 | 2000 | raw | 273.511 |
| ? | `mnist_dit_eps` | ddim | 50 | 10000 | ema | 672.388 |
| ? | `mnist_dit_eps` | ddim | 50 | 2000 | raw | 274.636 |
| ? | `mnist_dit_eps` | ancestral | 1000 | 10000 | ema | 33.177 |
| ? | `mnist_dit_rf` | euler | 4 | 10000 | ema | 141.250 |
| ? | `mnist_dit_rf` | euler | 8 | 10000 | ema | 34.777 |
| ? | `mnist_dit_rf` | euler | 16 | 10000 | ema | 12.980 |
| ? | `mnist_dit_rf` | euler | 32 | 10000 | ema | 7.029 |
| ? | `mnist_dit_rf` | euler | 64 | 10000 | ema | 4.992 |
| ? | `mnist_unet_eps` | ddim | 10 | 10000 | ema | 22.059 |
| ? | `mnist_unet_eps` | ddim | 20 | 10000 | ema | 4.572 |
| ? | `mnist_unet_eps` | ddim | 50 | 10000 | ema | 4.315 |
| ? | `mnist_unet_eps` | ancestral | 1000 | 10000 | ema | 2.790 |
| ? | `mnist_unet_rf` | euler | 4 | 10000 | ema | 96.610 |
| ? | `mnist_unet_rf` | euler | 8 | 10000 | ema | 18.735 |
| ? | `mnist_unet_rf` | euler | 16 | 10000 | ema | 5.059 |
| ? | `mnist_unet_rf` | euler | 32 | 10000 | ema | 1.787 |
| ? | `mnist_unet_rf` | euler | 64 | 10000 | ema | 0.877 |
