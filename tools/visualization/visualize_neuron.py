import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from drone_control.algorithms.ppo import ActorCritic, device
import os

class NeuronVisualizer:
    """Trực quan hóa kiến trúc mạng Actor và trọng số"""
    
    def __init__(self, model_path=None, state_dim=12, action_dim=4):
        """
        Args:
            model_path: Đường dẫn đến file model (.pth)
            state_dim: Kích thước state
            action_dim: Kích thước action
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Tạo model
        self.model = ActorCritic(state_dim, action_dim, action_std_init=0.6).to(device)
        
        # Load model nếu có
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✓ Đã load model từ: {model_path}")
        else:
            print("⚠ Sử dụng model mới (chưa train)")
        
        self.model.eval()
    
    def visualize_architecture(self, save_path=None):
        """Vẽ kiến trúc mạng actor"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Tham số vẽ
        layer_positions = [
            ('Input\nState', 0, self.state_dim),
            ('Hidden 1\n(Tanh)', 2, 64),
            ('Hidden 2\n(Tanh)', 4, 64),
            ('Output\nAction', 6, self.action_dim),
        ]
        
        x_positions = [pos[1] for pos in layer_positions]
        layer_names = [pos[0] for pos in layer_positions]
        neuron_counts = [pos[2] for pos in layer_positions]
        
        # Vẽ các neurons
        neuron_y_positions = []
        for layer_idx, (name, x, count) in enumerate(layer_positions):
            y_start = -count / 2
            y_positions = [y_start + i for i in range(count)]
            neuron_y_positions.append(y_positions)
            
            # Vẽ neuron circles
            for y in y_positions:
                circle = mpatches.Circle((x, y), 0.25, color='lightblue', ec='darkblue', linewidth=1.5)
                ax.add_patch(circle)
            
            # Vẽ tên layer
            ax.text(x, neuron_counts[layer_idx]/2 + 2, name, ha='center', fontsize=11, fontweight='bold')
        
        # Vẽ các kết nối (connections)
        # Layer 0 -> Layer 1
        self._draw_connections(ax, 0, 1, neuron_y_positions, 'weight')
        # Layer 1 -> Layer 2
        self._draw_connections(ax, 1, 2, neuron_y_positions, 'weight')
        # Layer 2 -> Layer 3
        self._draw_connections(ax, 2, 3, neuron_y_positions, 'weight')
        
        # Hiển thị số neurons
        for i, count in enumerate(neuron_counts):
            ax.text(x_positions[i], -count/2 - 2, f'{count} neurons', ha='center', fontsize=9, style='italic')
        
        ax.set_xlim(-1, 7)
        ax.set_ylim(-max(neuron_counts)/2 - 4, max(neuron_counts)/2 + 4)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Actor Network Architecture', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Đã lưu: {save_path}")
        plt.show()
    
    def _draw_connections(self, ax, layer_from, layer_to, neuron_y_positions, style='weight'):
        """Vẽ các kết nối giữa layers"""
        x_from = layer_from * 2
        x_to = layer_to * 2
        y_from_list = neuron_y_positions[layer_from]
        y_to_list = neuron_y_positions[layer_to]
        
        # Vẽ các connections với độ dày dựa trên trọng số
        for y_from in y_from_list:
            for y_to in y_to_list:
                # Màu xám nhẹ cho các kết nối
                ax.plot([x_from + 0.25, x_to - 0.25], [y_from, y_to], 
                       color='gray', alpha=0.3, linewidth=0.5, zorder=1)
    
    def visualize_weights_heatmap(self, save_path=None):
        """Vẽ biểu đồ nhiệt trọng số của các layers"""
        actor_layers = [layer for layer in self.model.actor.modules() if isinstance(layer, nn.Linear)]
        
        n_layers = len(actor_layers)
        fig, axes = plt.subplots(1, n_layers, figsize=(15, 4))
        if n_layers == 1:
            axes = [axes]
        
        for idx, layer in enumerate(actor_layers):
            weights = layer.weight.data.cpu().numpy()
            
            im = axes[idx].imshow(weights, cmap='RdBu_r', aspect='auto', vmin=-weights.std()*3, vmax=weights.std()*3)
            axes[idx].set_title(f'Layer {idx+1}\n({weights.shape[1]}→{weights.shape[0]})', fontsize=10, fontweight='bold')
            axes[idx].set_xlabel('Input neurons')
            axes[idx].set_ylabel('Output neurons')
            plt.colorbar(im, ax=axes[idx], label='Weight value')
        
        plt.suptitle('Actor Network - Weight Heatmaps', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Đã lưu: {save_path}")
        plt.show()
    
    def visualize_activations(self, state_sample=None, save_path=None):
        """Trực quan hóa kích hoạt (activations) khi đi qua mạng với một state mẫu"""
        if state_sample is None:
            # Tạo state mẫu
            state_sample = torch.randn(1, self.state_dim).to(device)
        elif isinstance(state_sample, np.ndarray):
            state_sample = torch.FloatTensor(state_sample).to(device)
        
        # Forward pass với hook để capture activations
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activations[name] = output.detach().cpu().numpy()
            return hook
        
        # Register hooks
        hooks = []
        for idx, module in enumerate(self.model.actor.modules()):
            if isinstance(module, (nn.Linear, nn.Tanh)):
                hook = module.register_forward_hook(hook_fn(f'layer_{idx}'))
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = self.model.actor(state_sample)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Vẽ activations
        filtered_activations = {k: v for k, v in activations.items() if len(v.shape) > 1}
        n_plots = len(filtered_activations)
        
        fig, axes = plt.subplots(2, (n_plots + 1) // 2, figsize=(14, 8))
        axes = axes.flatten()
        
        for idx, (name, act) in enumerate(filtered_activations.items()):
            if idx < len(axes):
                im = axes[idx].imshow(act.T, cmap='hot', aspect='auto')
                axes[idx].set_title(name, fontweight='bold')
                axes[idx].set_xlabel('Batch')
                axes[idx].set_ylabel('Neuron index')
                plt.colorbar(im, ax=axes[idx], label='Activation')
        
        # Ẩn các subplot không dùng
        for idx in range(len(filtered_activations), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Actor Network - Neuron Activations', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Đã lưu: {save_path}")
        plt.show()
    
    def visualize_weight_distribution(self, save_path=None):
        """Vẽ phân bố của trọng số"""
        actor_layers = [layer for layer in self.model.actor.modules() if isinstance(layer, nn.Linear)]
        
        fig, axes = plt.subplots(1, len(actor_layers), figsize=(14, 4))
        if len(actor_layers) == 1:
            axes = [axes]
        
        for idx, layer in enumerate(actor_layers):
            weights = layer.weight.data.cpu().numpy().flatten()
            
            axes[idx].hist(weights, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
            axes[idx].set_title(f'Layer {idx+1} Weights Distribution', fontweight='bold')
            axes[idx].set_xlabel('Weight value')
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(alpha=0.3)
            
            # Thêm thống kê
            axes[idx].text(0.98, 0.97, f'Mean: {weights.mean():.4f}\nStd: {weights.std():.4f}',
                          transform=axes[idx].transAxes, fontsize=9, verticalalignment='top',
                          horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Actor Network - Weight Distributions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Đã lưu: {save_path}")
        plt.show()
    
    def print_network_summary(self):
        """In thông tin tóm tắt về mạng"""
        print("\n" + "="*60)
        print("ACTOR NETWORK SUMMARY")
        print("="*60)
        print(f"Input dimension:  {self.state_dim}")
        print(f"Output dimension: {self.action_dim}")
        print("\nNetwork Structure:")
        print(self.model.actor)
        
        # Tính số parameters
        total_params = sum(p.numel() for p in self.model.actor.parameters())
        trainable_params = sum(p.numel() for p in self.model.actor.parameters() if p.requires_grad)
        
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("="*60 + "\n")


def main():
    """Hàm main để chạy visualization"""
    
    # Cấu hình
    STATE_DIM = 13  # State dimension từ trained model
    ACTION_DIM = 3  # Action dimension từ trained model
    
    # Tìm model mới nhất
    model_path = None
    if os.path.exists('log_dir/hover_projectairsim/ppo_hover_final.pth'):
        model_path = 'log_dir/hover_projectairsim/ppo_hover_final.pth'
    elif os.path.exists('log_dir/hover_projectairsim/ep2006_step100000.pth'):
        model_path = 'log_dir/hover_projectairsim/ep2006_step100000.pth'
    
    # Tạo visualizer
    print("🔧 Đang tải model...")
    visualizer = NeuronVisualizer(model_path=model_path, state_dim=STATE_DIM, action_dim=ACTION_DIM)
    
    # In thông tin mạng
    visualizer.print_network_summary()
    
    # Tạo các visualization
    print("📊 Đang tạo visualizations...")
    
    print("\n[1/4] Kiến trúc mạng...")
    visualizer.visualize_architecture(save_path='actor_architecture.png')
    
    print("[2/4] Biểu đồ nhiệt trọng số...")
    visualizer.visualize_weights_heatmap(save_path='actor_weights_heatmap.png')
    
    print("[3/4] Phân bố trọng số...")
    visualizer.visualize_weight_distribution(save_path='actor_weight_distribution.png')
    
    print("[4/4] Kích hoạt neuron...")
    # Sử dụng state mẫu
    sample_state = torch.randn(1, STATE_DIM).to(device)
    visualizer.visualize_activations(state_sample=sample_state, save_path='actor_activations.png')
    
    print("\n✅ Hoàn thành! Các hình ảnh đã được lưu trong thư mục hiện tại.")


if __name__ == '__main__':
    main()
