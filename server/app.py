from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from werkzeug.utils import secure_filename
import logging
from typing import Dict, List, Optional
import traceback

# Import your GA class (assuming it's in a separate file)
# from jit_ga_model import JITSupplyChainGA

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODELS_FOLDER'] = 'models'
app.config['RESULTS_FOLDER'] = 'results'

# Create necessary directories
for folder in [app.config['UPLOAD_FOLDER'], app.config['MODELS_FOLDER'], app.config['RESULTS_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
ga_model = None
model_info = {
    'loaded': False,
    'model_path': None,
    'loaded_at': None,
    'training_info': {}
}

class JITSupplyChainGA:
    """
    Simplified version of your GA class for the backend
    Include the core methods needed for inference
    """
    
    def __init__(self, population_size=100, generations=200, crossover_rate=0.8, 
                 mutation_rate=0.1, elitism_rate=0.1):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        
        # JIT-specific parameters
        self.jit_tolerance = 0.1
        self.holding_cost_factor = 0.05
        self.stockout_penalty = 1000
        self.late_delivery_penalty = 500
        
        # Model state
        self.best_solution = None
        self.fitness_history = []
        self.processed_data = None
        self.is_trained = False
    
    def load_model(self, filepath):
        """Load a trained GA model from disk"""
        try:
            with open(filepath, 'rb') as f:
                model_state = pickle.load(f)
            
            # Restore model state
            self.best_solution = model_state.get('best_solution')
            self.best_fitness = model_state.get('best_fitness')
            self.fitness_history = model_state.get('fitness_history', [])
            self.processed_data = model_state.get('processed_data')
            self.is_trained = model_state.get('optimization_completed', False)
            
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def predict_optimization(self, new_data):
        """Apply the trained model to new data"""
        if not self.is_trained or self.best_solution is None:
            raise ValueError("Model not trained or loaded")
        
        # Apply the learned optimization patterns to new data
        recommendations = {}
        
        for product in new_data['Product Name'].unique():
            product_data = new_data[new_data['Product Name'] == product]
            
            if product in self.best_solution:
                # Use learned solution
                solution = self.best_solution[product]
                recommendations[product] = {
                    'recommended_production': solution['production_quantity'],
                    'recommended_inventory': solution['inventory_level'],
                    'recommended_fulfillment_days': solution['planned_fulfillment_days'],
                    'safety_stock': solution['safety_stock'],
                    'confidence': 'high'
                }
            else:
                # Apply patterns from similar products
                avg_demand = product_data['Order Quantity'].mean() if 'Order Quantity' in product_data.columns else 100
                recommendations[product] = {
                    'recommended_production': int(avg_demand * 1.1),
                    'recommended_inventory': int(avg_demand * 0.3),
                    'recommended_fulfillment_days': 5,
                    'safety_stock': int(avg_demand * 0.1),
                    'confidence': 'medium'
                }
        
        return recommendations

# Helper functions
def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx', 'xls'}

def load_ga_model(model_path):
    """Load the GA model"""
    global ga_model, model_info
    
    try:
        ga_model = JITSupplyChainGA()
        success = ga_model.load_model(model_path)
        
        if success:
            model_info.update({
                'loaded': True,
                'model_path': model_path,
                'loaded_at': datetime.now().isoformat(),
                'training_info': {
                    'best_fitness': getattr(ga_model, 'best_fitness', None),
                    'generations_trained': len(getattr(ga_model, 'fitness_history', [])),
                    'population_size': ga_model.population_size
                }
            })
            logger.info(f"Model loaded successfully from {model_path}")
            return True
        else:
            model_info['loaded'] = False
            return False
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        model_info['loaded'] = False
        return False

def create_visualization(data, chart_type):
    """Create visualization and return as base64 encoded image"""
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    try:
        if chart_type == 'production_vs_demand':
            products = list(data.keys())[:10]  # Top 10 products
            production = [data[p]['recommended_production'] for p in products]
            
            ax.bar(range(len(products)), production, alpha=0.7)
            ax.set_title('Recommended Production Quantities')
            ax.set_xlabel('Products')
            ax.set_ylabel('Quantity')
            ax.set_xticks(range(len(products)))
            ax.set_xticklabels([p[:10] + '...' if len(p) > 10 else p for p in products], 
                             rotation=45, ha='right')
            
        elif chart_type == 'inventory_distribution':
            inventory_levels = [data[p]['recommended_inventory'] for p in data.keys()]
            ax.hist(inventory_levels, bins=15, alpha=0.7, color='green')
            ax.set_title('Distribution of Recommended Inventory Levels')
            ax.set_xlabel('Inventory Level')
            ax.set_ylabel('Frequency')
            
        elif chart_type == 'fulfillment_times':
            fulfillment_times = [data[p]['recommended_fulfillment_days'] for p in data.keys()]
            ax.hist(fulfillment_times, bins=10, alpha=0.7, color='orange')
            ax.set_title('Distribution of Recommended Fulfillment Times')
            ax.set_xlabel('Fulfillment Days')
            ax.set_ylabel('Frequency')
        
        plt.tight_layout()
        
        # Save to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return img_base64
        
    except Exception as e:
        plt.close()
        logger.error(f"Error creating visualization: {str(e)}")
        return None

# API Endpoints

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model_info['loaded']
    })

@app.route('/api/model/info', methods=['GET'])
def get_model_info():
    """Get information about the loaded model"""
    return jsonify(model_info)

@app.route('/api/model/load', methods=['POST'])
def load_model():
    """Load a trained GA model"""
    try:
        data = request.get_json()
        model_path = data.get('model_path')
        
        if not model_path or not os.path.exists(model_path):
            return jsonify({
                'success': False,
                'error': 'Model file not found'
            }), 400
        
        success = load_ga_model(model_path)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Model loaded successfully',
                'model_info': model_info
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to load model'
            }), 500
            
    except Exception as e:
        logger.error(f"Error in load_model endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload CSV files for optimization"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            return jsonify({
                'success': True,
                'message': 'File uploaded successfully',
                'filename': filename,
                'filepath': filepath
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Please upload CSV or Excel files.'
            }), 400
            
    except Exception as e:
        logger.error(f"Error in upload endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/optimize', methods=['POST'])
def optimize_supply_chain():
    """Run supply chain optimization on uploaded data"""
    global ga_model
    
    try:
        if not model_info['loaded'] or ga_model is None:
            return jsonify({
                'success': False,
                'error': 'No model loaded. Please load a trained model first.'
            }), 400
        
        data = request.get_json()
        file_path = data.get('file_path')
        
        if not file_path or not os.path.exists(file_path):
            return jsonify({
                'success': False,
                'error': 'Data file not found'
            }), 400
        
        # Load and process the data
        df = pd.read_csv(file_path)
        
        # Ensure required columns exist
        required_columns = ['Product Name']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return jsonify({
                'success': False,
                'error': f'Missing required columns: {missing_columns}'
            }), 400
        
        # Add default columns if missing
        if 'Order Quantity' not in df.columns:
            df['Order Quantity'] = 100  # Default quantity
        
        # Run optimization
        recommendations = ga_model.predict_optimization(df)
        
        # Calculate summary statistics
        total_products = len(recommendations)
        total_production = sum(r['recommended_production'] for r in recommendations.values())
        total_inventory = sum(r['recommended_inventory'] for r in recommendations.values())
        avg_fulfillment = np.mean([r['recommended_fulfillment_days'] for r in recommendations.values()])
        
        # Create visualizations
        charts = {}
        chart_types = ['production_vs_demand', 'inventory_distribution', 'fulfillment_times']
        
        for chart_type in chart_types:
            chart_data = create_visualization(recommendations, chart_type)
            if chart_data:
                charts[chart_type] = chart_data
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(app.config['RESULTS_FOLDER'], f"optimization_results_{timestamp}.json")
        
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_products': total_products,
                'total_recommended_production': int(total_production),
                'total_recommended_inventory': int(total_inventory),
                'average_fulfillment_days': round(avg_fulfillment, 2)
            },
            'recommendations': recommendations,
            'model_info': model_info
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        return jsonify({
            'success': True,
            'message': 'Optimization completed successfully',
            'results': results_data,
            'charts': charts,
            'results_file': results_file
        })
        
    except Exception as e:
        logger.error(f"Error in optimize endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/optimize/batch', methods=['POST'])
def batch_optimize():
    """Run batch optimization on multiple files"""
    try:
        if not model_info['loaded'] or ga_model is None:
            return jsonify({
                'success': False,
                'error': 'No model loaded'
            }), 400
        
        data = request.get_json()
        file_paths = data.get('file_paths', [])
        
        if not file_paths:
            return jsonify({
                'success': False,
                'error': 'No files provided'
            }), 400
        
        batch_results = []
        
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    recommendations = ga_model.predict_optimization(df)
                    
                    batch_results.append({
                        'file': os.path.basename(file_path),
                        'status': 'success',
                        'products_optimized': len(recommendations),
                        'recommendations': recommendations
                    })
                else:
                    batch_results.append({
                        'file': os.path.basename(file_path),
                        'status': 'error',
                        'error': 'File not found'
                    })
                    
            except Exception as e:
                batch_results.append({
                    'file': os.path.basename(file_path),
                    'status': 'error',
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'message': 'Batch optimization completed',
            'results': batch_results
        })
        
    except Exception as e:
        logger.error(f"Error in batch optimize endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/results/<filename>', methods=['GET'])
def get_results(filename):
    """Download optimization results"""
    try:
        file_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({
                'success': False,
                'error': 'Results file not found'
            }), 404
            
    except Exception as e:
        logger.error(f"Error in get_results endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/results/list', methods=['GET'])
def list_results():
    """List all available results files"""
    try:
        results_files = []
        results_dir = app.config['RESULTS_FOLDER']
        
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(results_dir, filename)
                file_stat = os.stat(file_path)
                
                results_files.append({
                    'filename': filename,
                    'size': file_stat.st_size,
                    'created': datetime.fromtimestamp(file_stat.st_ctime).isoformat()
                })
        
        return jsonify({
            'success': True,
            'files': sorted(results_files, key=lambda x: x['created'], reverse=True)
        })
        
    except Exception as e:
        logger.error(f"Error listing results: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict_single():
    """Make prediction for a single product or scenario"""
    try:
        if not model_info['loaded'] or ga_model is None:
            return jsonify({
                'success': False,
                'error': 'No model loaded'
            }), 400
        
        data = request.get_json()
        product_name = data.get('product_name')
        order_quantity = data.get('order_quantity', 100)
        
        if not product_name:
            return jsonify({
                'success': False,
                'error': 'Product name is required'
            }), 400
        
        # Create a simple dataframe for single prediction
        df = pd.DataFrame({
            'Product Name': [product_name],
            'Order Quantity': [order_quantity]
        })
        
        recommendations = ga_model.predict_optimization(df)
        
        return jsonify({
            'success': True,
            'recommendation': recommendations.get(product_name, {}),
            'product_name': product_name
        })
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Auto-load model if available
    default_model_path = 'models/jit_ga_model.pkl'
    if os.path.exists(default_model_path):
        load_ga_model(default_model_path)
        logger.info("Default model loaded automatically")
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    app.run(host='0.0.0.0', port=port, debug=debug)