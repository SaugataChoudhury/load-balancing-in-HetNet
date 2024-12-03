import numpy as np
from typing import List, Dict, Tuple
from enum import Enum

class NodeType(Enum):
    MACRO_CELL = 1
    SMALL_CELL = 2
    FEMTO_CELL = 3

class UserEquipment:
    def __init__(self, user_id: str, lat: float, long: float, bandwidth_demand: float, mobility_class: str):
        """
        Represents a user equipment with network requirements.
        
        Args:
        - user_id: Unique identifier for the user
        - lat, long: Geographical coordinates
        - bandwidth_demand: Expected data rate requirements
        - mobility_class: User's mobility characteristics
        """
        self.user_id = user_id
        self.lat = lat
        self.long = long
        self.bandwidth_demand = bandwidth_demand
        self.mobility_class = mobility_class  # 'high', 'medium', 'low'

class NetworkNode:
    def __init__(self, node_id: str, node_type: NodeType, capacity: float, lat: float, long: float):
        """
        Represents a network node in 5G HetNet.
        
        Args:
        - node_id: Unique identifier for the node
        - node_type: Type of network node
        - capacity: Maximum supported bandwidth
        - lat, long: Node's geographical location
        """
        self.node_id = node_id
        self.type = node_type
        self.max_capacity = capacity
        self.current_load = 0
        self.lat = lat
        self.long = long
        self.connected_users = []
    
    def can_support_user(self, user: UserEquipment) -> bool:
        """
        Check if node can support user's bandwidth requirements.
        
        Considers:
        - Remaining node capacity
        - User's bandwidth demand
        - Distance-based signal quality
        """
        distance = self.calculate_distance(user)
        signal_quality_factor = self.get_signal_quality_factor(distance)
        
        return (self.current_load + user.bandwidth_demand * signal_quality_factor) <= self.max_capacity

    def calculate_distance(self, user: UserEquipment) -> float:
        """Calculate geographical distance between node and user."""
        return np.sqrt(
            (self.lat - user.lat)**2 + 
            (self.long - user.long)**2
        )

    def get_signal_quality_factor(self, distance: float) -> float:
        """
        Calculate signal quality attenuation based on distance.
        Provides exponential decay of signal strength.
        """
        return max(0.1, np.exp(-0.1 * distance))

class PredictiveHetNetLoadBalancer:
    def __init__(self, prediction_model):
        """
        5G HetNet Load Balancer with Predictive Capabilities.
        
        Args:
        - prediction_model: ML model predicting user distribution
        """
        self.network_nodes: List[NetworkNode] = []
        self.prediction_model = prediction_model
    
    def add_network_node(self, node: NetworkNode):
        """Add a network node to the balancer's management."""
        self.network_nodes.append(node)
    
    def predict_user_distribution(self, timestamp):
        """
        Use ML model to predict user distribution.
        
        Returns list of predicted UserEquipment instances.
        """
        # Replace with actual model prediction logic
        return self.prediction_model.predict(timestamp)
    
    def balance_load(self, timestamp):
        """
        Primary load balancing method.
        Redistributes users across network nodes.
        """
        # Reset node loads
        for node in self.network_nodes:
            node.current_load = 0
            node.connected_users = []
        
        # Predict user distribution
        predicted_users = self.predict_user_distribution(timestamp)
        
        # Sort nodes by current load to enable intelligent distribution
        sorted_nodes = sorted(
            self.network_nodes, 
            key=lambda node: node.current_load
        )
        
        # Prioritize small/femto cells for targeted load distribution
        small_femto_cells = [
            node for node in sorted_nodes 
            if node.type in [NodeType.SMALL_CELL, NodeType.FEMTO_CELL]
        ]
        macro_cells = [
            node for node in sorted_nodes 
            if node.type == NodeType.MACRO_CELL
        ]
        
        # First pass: distribute to small/femto cells
        for user in predicted_users:
            assigned = False
            for node in small_femto_cells:
                if node.can_support_user(user):
                    node.connected_users.append(user)
                    node.current_load += user.bandwidth_demand
                    assigned = True
                    break
            
            # Fallback to macro cells if no small cell available
            if not assigned:
                for node in macro_cells:
                    if node.can_support_user(user):
                        node.connected_users.append(user)
                        node.current_load += user.bandwidth_demand
                        break
    
    def get_load_statistics(self):
        """
        Generate load distribution statistics.
        
        Returns:
        - Average load percentage
        - Node-wise load details
        """
        total_load = sum(node.current_load for node in self.network_nodes)
        total_capacity = sum(node.max_capacity for node in self.network_nodes)
        
        return {
            'overall_load_percentage': (total_load / total_capacity) * 100,
            'node_details': [
                {
                    'node_id': node.node_id,
                    'type': node.type.name,
                    'current_load': node.current_load,
                    'max_capacity': node.max_capacity,
                    'load_percentage': (node.current_load / node.max_capacity) * 100,
                    'connected_users': len(node.connected_users)
                } for node in self.network_nodes
            ]
        }

# Example usage and initialization
def main():
    # Simulated ML prediction model (replace with actual model)
    class SimplePredictionModel:
        def predict(self, timestamp):
            # Generate synthetic user distribution
            return [
                UserEquipment(
                    user_id=f'user_{i}', 
                    lat=np.random.uniform(0, 90),
                    long=np.random.uniform(0, 180),
                    bandwidth_demand=np.random.uniform(1, 10),
                    mobility_class='medium'
                ) for i in range(100)
            ]
    
    # Initialize load balancer
    prediction_model = SimplePredictionModel()
    lb = PredictiveHetNetLoadBalancer(prediction_model)
    
    # Add network nodes
    lb.add_network_node(NetworkNode(
        node_id='macro_1', 
        node_type=NodeType.MACRO_CELL, 
        capacity=100, 
        lat=40.7128, 
        long=-74.0060
    ))
    
    lb.add_network_node(NetworkNode(
        node_id='small_1', 
        node_type=NodeType.SMALL_CELL, 
        capacity=20, 
        lat=40.7150, 
        long=-74.0070
    ))
    
    # Perform load balancing
    lb.balance_load(timestamp='2024-01-01 12:00:00')
    
    # Get and print load statistics
    stats = lb.get_load_statistics()
    print(f"Overall Network Load: {stats['overall_load_percentage']:.2f}%")
    for node in stats['node_details']:
        print(f"Node {node['node_id']} Load: {node['load_percentage']:.2f}%")

if __name__ == "__main__":
    main()