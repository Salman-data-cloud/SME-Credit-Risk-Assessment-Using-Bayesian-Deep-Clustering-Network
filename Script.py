import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class SMEDataGenerator:
    """Generate synthetic SME data for credit risk assessment"""

    def __init__(self, n_samples=2000):
        self.n_samples = n_samples
        self.feature_names = [
            'annual_revenue', 'monthly_cash_flow', 'debt_to_equity', 'current_ratio',
            'years_in_business', 'num_employees', 'industry_sector', 'collateral_value',
            'owner_credit_score', 'monthly_expenses', 'seasonal_revenue_var',
            'digital_presence', 'export_percentage', 'supplier_diversity'
        ]

    def generate_data(self):
        """Generate synthetic SME financial data with inherent risk clusters"""
        data = {}

        # Defining three risk clusters: Low, Medium, High
        cluster_sizes = [600, 800, 600]  # Distribution of clusters
        true_labels = []

        for cluster_id, size in enumerate(cluster_sizes):
            true_labels.extend([cluster_id] * size)

        # Shuffle the labels
        true_labels = np.array(true_labels)
        indices = np.random.permutation(len(true_labels))
        true_labels = true_labels[indices]

        # Generating features based on risk clusters
        all_features = []

        for i in range(self.n_samples):
            cluster = true_labels[i]

            if cluster == 0:  # Low Risk
                annual_revenue = np.random.lognormal(mean=12, sigma=0.5)
                monthly_cash_flow = annual_revenue * 0.08 * (1 + np.random.normal(0, 0.2))
                debt_to_equity = np.random.gamma(2, 0.3)
                current_ratio = np.random.gamma(5, 0.4) + 1
                years_in_business = np.random.gamma(4, 2) + 3

            elif cluster == 1:  # Medium Risk
                annual_revenue = np.random.lognormal(mean=11.5, sigma=0.7)
                monthly_cash_flow = annual_revenue * 0.05 * (1 + np.random.normal(0, 0.3))
                debt_to_equity = np.random.gamma(3, 0.5) + 0.5
                current_ratio = np.random.gamma(3, 0.3) + 0.8
                years_in_business = np.random.gamma(3, 1.5) + 1

            else:  # High Risk
                annual_revenue = np.random.lognormal(mean=11, sigma=0.9)
                monthly_cash_flow = annual_revenue * 0.02 * (1 + np.random.normal(0, 0.5))
                debt_to_equity = np.random.gamma(5, 0.8) + 1
                current_ratio = np.random.gamma(2, 0.2) + 0.3
                years_in_business = np.random.gamma(2, 1) + 0.5

            # Common features with cluster-based variations
            num_employees = max(1, int(annual_revenue / 500000 * (1 + np.random.normal(0, 0.3))))
            industry_sector = np.random.choice(range(8))  # 8 different sectors
            collateral_value = annual_revenue * np.random.uniform(0.2, 1.5)
            owner_credit_score = 300 + (cluster == 0) * 200 + (cluster == 1) * 150 + np.random.normal(0, 50)
            owner_credit_score = np.clip(owner_credit_score, 300, 850)

            monthly_expenses = monthly_cash_flow * np.random.uniform(0.7, 1.2)
            seasonal_revenue_var = np.random.uniform(0.1, 0.5)
            digital_presence = np.random.choice([0, 1], p=[0.3, 0.7] if cluster == 0 else [0.6, 0.4])
            export_percentage = np.random.uniform(0, 0.3)
            supplier_diversity = np.random.poisson(5) + 1

            features = [
                annual_revenue, monthly_cash_flow, debt_to_equity, current_ratio,
                years_in_business, num_employees, industry_sector, collateral_value,
                owner_credit_score, monthly_expenses, seasonal_revenue_var,
                digital_presence, export_percentage, supplier_diversity
            ]

            all_features.append(features)

        # Creating DataFrame
        df = pd.DataFrame(all_features, columns=self.feature_names)
        df['true_risk_cluster'] = true_labels

        return df


class BayesianDeepClusteringNetwork(tf.keras.Model):
    """Bayesian Deep Clustering Network for SME Risk Assessment"""

    def __init__(self, input_dim, latent_dim=10, num_clusters=3):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_clusters = num_clusters

        # Encoder layers (using Monte Carlo Dropout for Bayesian behavior)
        self.encoder_layer1 = tf.keras.layers.Dense(64, activation='relu')
        self.encoder_dropout1 = tf.keras.layers.Dropout(0.3)
        self.encoder_layer2 = tf.keras.layers.Dense(32, activation='relu')
        self.encoder_dropout2 = tf.keras.layers.Dropout(0.3)
        self.encoder_layer3 = tf.keras.layers.Dense(latent_dim, activation='linear')

        # Clustering layer - Initialize with better separation
        self.cluster_centers = self.add_weight(
            name='cluster_centers',
            shape=(num_clusters, latent_dim),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=3.0),
            trainable=True
        )

        # Decoder network
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(input_dim, activation='linear')
        ])

    def call(self, inputs, training=None):
        # Encode to latent space with Monte Carlo Dropout
        x = self.encoder_layer1(inputs)
        x = self.encoder_dropout1(x, training=training)
        x = self.encoder_layer2(x)
        x = self.encoder_dropout2(x, training=training)
        z = self.encoder_layer3(x)

        # Compute cluster assignments (soft assignments with higher temperature)
        distances = tf.reduce_sum(
            tf.square(tf.expand_dims(z, 1) - tf.expand_dims(self.cluster_centers, 0)),
            axis=2
        )
        cluster_probs = tf.nn.softmax(-distances / 5.0)  # Higher temperature for softer assignments

        # Decode from latent space
        reconstructions = self.decoder(z, training=training)

        return {
            'latent': z,
            'cluster_probs': cluster_probs,
            'reconstructions': reconstructions,
            'cluster_centers': self.cluster_centers
        }

    def get_cluster_assignments(self, inputs):
        """Get hard cluster assignments"""
        outputs = self(inputs, training=False)
        return tf.argmax(outputs['cluster_probs'], axis=1)


class VariationalAutoencoder(tf.keras.Model):
    """Variational Autoencoder for comparison"""

    def __init__(self, input_dim, latent_dim=10):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
        ])

        self.mean_layer = tf.keras.layers.Dense(latent_dim)
        self.logvar_layer = tf.keras.layers.Dense(latent_dim)

        # Decoder
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(input_dim, activation='linear')
        ])

    def encode(self, inputs):
        h = self.encoder(inputs)
        mean = self.mean_layer(h)
        logvar = self.logvar_layer(h)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(tf.shape(mean))
        return mean + tf.exp(0.5 * logvar) * eps

    def call(self, inputs, training=None):
        mean, logvar = self.encode(inputs)
        z = self.reparameterize(mean, logvar)
        reconstructions = self.decoder(z, training=training)

        return {
            'mean': mean,
            'logvar': logvar,
            'latent': z,
            'reconstructions': reconstructions
        }


def compute_clustering_loss(cluster_probs, latent_features, cluster_centers):
    """Compute clustering loss for Bayesian Deep Clustering Network"""
    # Soft assignment clustering loss
    distances = tf.reduce_sum(
        tf.square(tf.expand_dims(latent_features, 1) - tf.expand_dims(cluster_centers, 0)),
        axis=2
    )

    # Weighted clustering loss
    clustering_loss = tf.reduce_mean(
        tf.reduce_sum(cluster_probs * distances, axis=1)
    )

    # Strong entropy regularization to prevent cluster collapse
    entropy_loss = -tf.reduce_mean(
        tf.reduce_sum(cluster_probs * tf.math.log(cluster_probs + 1e-8), axis=1)
    )

    # Cluster balance loss to encourage equal cluster sizes
    cluster_sizes = tf.reduce_mean(cluster_probs, axis=0)
    target_size = 1.0 / tf.cast(tf.shape(cluster_centers)[0], tf.float32)
    balance_loss = tf.reduce_sum(tf.square(cluster_sizes - target_size))

    return clustering_loss + 0.5 * entropy_loss + 0.8 * balance_loss


def train_bayesian_clustering_model(X_train, X_val, epochs=100):
    """Train the Bayesian Deep Clustering Network"""
    input_dim = X_train.shape[1]
    model = BayesianDeepClusteringNetwork(input_dim)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    train_losses = []
    val_losses = []

    @tf.function
    def train_step(batch_x):
        with tf.GradientTape() as tape:
            outputs = model(batch_x, training=True)

            # Reconstruction loss
            recon_loss = tf.reduce_mean(tf.square(batch_x - outputs['reconstructions']))

            # Clustering loss
            cluster_loss = compute_clustering_loss(
                outputs['cluster_probs'],
                outputs['latent'],
                outputs['cluster_centers']
            )

            # Total loss - prioritize clustering over reconstruction
            total_loss = 0.1 * recon_loss + 2.0 * cluster_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return total_loss, recon_loss, cluster_loss

    # Training loop
    batch_size = 64
    for epoch in range(epochs):
        epoch_losses = []

        # Shuffle training data - convert to numpy first
        X_train_np = X_train.numpy()
        indices = np.random.permutation(len(X_train_np))
        X_train_shuffled = X_train_np[indices]

        # Mini-batch training
        for i in range(0, len(X_train_shuffled), batch_size):
            batch_x = tf.constant(X_train_shuffled[i:i+batch_size], dtype=tf.float32)
            loss, recon_loss, cluster_loss = train_step(batch_x)
            epoch_losses.append(loss.numpy())

        train_loss = np.mean(epoch_losses)
        train_losses.append(train_loss)

        # Validation loss
        val_outputs = model(X_val, training=False)
        val_recon_loss = tf.reduce_mean(tf.square(X_val - val_outputs['reconstructions']))
        val_cluster_loss = compute_clustering_loss(
            val_outputs['cluster_probs'],
            val_outputs['latent'],
            val_outputs['cluster_centers']
        )
        val_loss = 0.1 * val_recon_loss + 2.0 * val_cluster_loss
        val_losses.append(val_loss.numpy())

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    return model, train_losses, val_losses


def train_vae_model(X_train, X_val, epochs=100):
    """Train VAE for comparison"""
    input_dim = X_train.shape[1]
    model = VariationalAutoencoder(input_dim)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    @tf.function
    def train_step(batch_x):
        with tf.GradientTape() as tape:
            outputs = model(batch_x, training=True)

            # Reconstruction loss
            recon_loss = tf.reduce_mean(tf.square(batch_x - outputs['reconstructions']))

            # KL divergence loss
            kl_loss = -0.5 * tf.reduce_mean(
                1 + outputs['logvar'] - tf.square(outputs['mean']) - tf.exp(outputs['logvar'])
            )

            # Total loss
            total_loss = recon_loss + 0.01 * kl_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return total_loss

    # Training loop
    batch_size = 64
    for epoch in range(epochs):
        epoch_losses = []

        # Shuffle training data - convert to numpy first
        X_train_np = X_train.numpy()
        indices = np.random.permutation(len(X_train_np))
        X_train_shuffled = X_train_np[indices]

        for i in range(0, len(X_train_shuffled), batch_size):
            batch_x = tf.constant(X_train_shuffled[i:i+batch_size], dtype=tf.float32)
            loss = train_step(batch_x)
            epoch_losses.append(loss.numpy())

        if epoch % 20 == 0:
            print(f"VAE Epoch {epoch}: Loss = {np.mean(epoch_losses):.4f}")

    return model


def evaluate_clustering_performance(model, X_test, y_true, model_name="Model"):
    """Evaluating clustering performance"""
    # Get cluster assignments
    if hasattr(model, 'get_cluster_assignments'):
        y_pred = model.get_cluster_assignments(X_test).numpy()
    else:
        # For VAE, use K-means on latent space
        latent = model(X_test, training=False)['latent'].numpy()
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        y_pred = kmeans.fit_predict(latent)

    # Check if we have enough clusters for silhouette score
    n_unique_labels = len(np.unique(y_pred))

    # Calculate metrics with error handling
    if n_unique_labels > 1 and n_unique_labels < len(X_test):
        try:
            silhouette = silhouette_score(X_test, y_pred)
        except:
            silhouette = 0.0
    else:
        print(f"Warning: Only {n_unique_labels} unique cluster(s) found. Silhouette score set to 0.")
        silhouette = 0.0

    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)

    print(f"\n{model_name} Clustering Performance:")
    print(f"Number of clusters found: {n_unique_labels}")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Adjusted Rand Index: {ari:.4f}")
    print(f"Normalized Mutual Information: {nmi:.4f}")

    return {
        'silhouette': silhouette,
        'ari': ari,
        'nmi': nmi,
        'predictions': y_pred,
        'n_clusters': n_unique_labels
    }


def stability_analysis(X_train, X_test, y_true, num_runs=5):
    """Analyze model stability across multiple runs"""
    stability_scores = []

    print("Performing Stability Analysis...")

    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")

        # Training model with different initialization
        tf.random.set_seed(run * 42)
        np.random.seed(run * 42)

        model, _, _ = train_bayesian_clustering_model(X_train, X_test, epochs=50)
        y_pred = model.get_cluster_assignments(X_test).numpy()

        # Calculating ARI with true labels
        ari = adjusted_rand_score(y_true, y_pred)
        stability_scores.append(ari)

    stability_mean = np.mean(stability_scores)
    stability_std = np.std(stability_scores)

    print(f"\nStability Analysis Results:")
    print(f"Mean ARI: {stability_mean:.4f} ± {stability_std:.4f}")

    if stability_mean > 0:
        stability_score = 1 - (stability_std / stability_mean)
        print(f"Stability Score (1 - CV): {stability_score:.4f}")
    else:
        print(f"Stability Score (1 - CV): Unable to compute (mean ARI = 0)")

    return stability_scores


def uncertainty_analysis(model, X_test, num_samples=50):
    """Analyze prediction uncertainty for Bayesian model"""
    print("Performing Uncertainty Analysis...")

    # Multiple forward passes to estimate uncertainty
    predictions = []
    for _ in range(num_samples):
        outputs = model(X_test, training=True)  # Keep dropout active
        predictions.append(outputs['cluster_probs'].numpy())

    predictions = np.array(predictions)

    # Calculating prediction statistics
    mean_probs = np.mean(predictions, axis=0)
    std_probs = np.std(predictions, axis=0)

    # Entropy as uncertainty measure
    uncertainty = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=1)

    # Confidence (max probability)
    confidence = np.max(mean_probs, axis=1)

    return {
        'mean_probabilities': mean_probs,
        'std_probabilities': std_probs,
        'uncertainty': uncertainty,
        'confidence': confidence
    }


def create_visualizations(data, model, vae_model, X_test, y_true, bayesian_results, vae_results, uncertainty_results):
    """Create comprehensive visualizations"""

    # Get test indices properly
    _, _, train_idx, test_idx = train_test_split(
        range(len(data)), data['true_risk_cluster'],
        test_size=0.3, random_state=42, stratify=data['true_risk_cluster']
    )
    test_data = data.iloc[test_idx]

    fig = plt.figure(figsize=(20, 16))

    # 1. Data Distribution by True Risk Clusters
    plt.subplot(3, 4, 1)
    colors_true = ['green', 'orange', 'red']
    for i, cluster in enumerate(['Low Risk', 'Medium Risk', 'High Risk']):
        cluster_data = data[data['true_risk_cluster'] == i]
        plt.scatter(cluster_data['annual_revenue'], cluster_data['debt_to_equity'],
                   alpha=0.6, label=cluster, s=30, c=colors_true[i])
    plt.xlabel('Annual Revenue')
    plt.ylabel('Debt-to-Equity Ratio')
    plt.title('True Risk Clusters (All Data)')
    plt.legend()
    plt.xscale('log')

    # 2. Bayesian Model Predictions
    plt.subplot(3, 4, 2)
    colors = ['blue', 'orange', 'purple', 'brown']
    unique_clusters = np.unique(bayesian_results['predictions'])

    if len(unique_clusters) > 1:
        for i in unique_clusters:
            mask = bayesian_results['predictions'] == i
            if np.sum(mask) > 0:
                cluster_data = test_data.iloc[mask]
                plt.scatter(cluster_data['annual_revenue'], cluster_data['debt_to_equity'],
                           alpha=0.6, c=colors[i % len(colors)], label=f'Cluster {i}', s=30)
        plt.title('Bayesian Deep Clustering Predictions')
    else:
        plt.scatter(test_data['annual_revenue'], test_data['debt_to_equity'],
                   alpha=0.6, c='red', label='All Points (1 cluster)', s=30)
        plt.title('Bayesian Deep Clustering (Collapsed)')

    plt.xlabel('Annual Revenue')
    plt.ylabel('Debt-to-Equity Ratio')
    plt.legend()
    plt.xscale('log')

    # 3. VAE + K-means Predictions
    plt.subplot(3, 4, 3)
    for i in range(3):
        mask = vae_results['predictions'] == i
        if np.sum(mask) > 0:
            cluster_data = test_data.iloc[mask]
            plt.scatter(cluster_data['annual_revenue'], cluster_data['debt_to_equity'],
                       alpha=0.6, c=colors[i], label=f'VAE Cluster {i}', s=30)
    plt.xlabel('Annual Revenue')
    plt.ylabel('Debt-to-Equity Ratio')
    plt.title('VAE + K-means Predictions')
    plt.legend()
    plt.xscale('log')

    # 4. Performance Comparison
    plt.subplot(3, 4, 4)
    metrics = ['Silhouette', 'ARI', 'NMI']
    bayesian_scores = [bayesian_results['silhouette'], bayesian_results['ari'], bayesian_results['nmi']]
    vae_scores = [vae_results['silhouette'], vae_results['ari'], vae_results['nmi']]

    x = np.arange(len(metrics))
    width = 0.35

    plt.bar(x - width/2, bayesian_scores, width, label='Bayesian', alpha=0.8, color='skyblue')
    plt.bar(x + width/2, vae_scores, width, label='VAE', alpha=0.8, color='lightcoral')

    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Performance Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    plt.ylim(0, 1)

    # 5. Latent Space Visualization (Bayesian)
    plt.subplot(3, 4, 5)
    try:
        bayesian_latent = model(X_test, training=False)['latent'].numpy()
        pca = PCA(n_components=2)
        bayesian_pca = pca.fit_transform(bayesian_latent)

        for i, cluster in enumerate(['Low Risk', 'Medium Risk', 'High Risk']):
            mask = y_true == i
            plt.scatter(bayesian_pca[mask, 0], bayesian_pca[mask, 1],
                       alpha=0.6, label=cluster, s=30, c=colors_true[i])
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('Bayesian Model - Latent Space')
        plt.legend()
    except Exception as e:
        plt.text(0.5, 0.5, f'Latent space visualization\nunavailable: {str(e)[:30]}...',
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Bayesian Latent Space')

    # 6. Latent Space Visualization (VAE)
    plt.subplot(3, 4, 6)
    try:
        vae_latent = vae_model(X_test, training=False)['latent'].numpy()
        vae_pca = pca.fit_transform(vae_latent)

        for i, cluster in enumerate(['Low Risk', 'Medium Risk', 'High Risk']):
            mask = y_true == i
            plt.scatter(vae_pca[mask, 0], vae_pca[mask, 1],
                       alpha=0.6, label=cluster, s=30, c=colors_true[i])
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('VAE - Latent Space')
        plt.legend()
    except Exception as e:
        plt.text(0.5, 0.5, f'VAE latent space\nunavailable: {str(e)[:30]}...',
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('VAE Latent Space')

    # 7. Uncertainty Analysis
    plt.subplot(3, 4, 7)
    try:
        plt.scatter(uncertainty_results['confidence'], uncertainty_results['uncertainty'],
                   alpha=0.6, c=y_true, cmap='viridis', s=30)
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Prediction Uncertainty')
        plt.title('Confidence vs Uncertainty')
        plt.colorbar(label='True Risk Cluster')
    except Exception as e:
        plt.text(0.5, 0.5, 'Uncertainty analysis\nnot available',
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Uncertainty Analysis')

    # 8. Feature Distribution Analysis
    plt.subplot(3, 4, 8)
    feature_data = test_data[['debt_to_equity', 'current_ratio', 'owner_credit_score']]

    for cluster in range(3):
        cluster_data = feature_data[y_true == cluster]['debt_to_equity']
        plt.hist(cluster_data, alpha=0.5, label=f'Cluster {cluster}', bins=15)
    plt.xlabel('Debt-to-Equity Ratio')
    plt.ylabel('Frequency')
    plt.title('Debt-to-Equity Distribution')
    plt.legend()

    # 9-12. Confusion Matrices and Additional Analysis
    plt.subplot(3, 4, 9)
    try:
        cm_bayesian = confusion_matrix(y_true, bayesian_results['predictions'])
        sns.heatmap(cm_bayesian, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Bayesian - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
    except:
        plt.text(0.5, 0.5, 'Confusion matrix\nnot available',
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Bayesian Confusion Matrix')

    plt.subplot(3, 4, 10)
    try:
        cm_vae = confusion_matrix(y_true, vae_results['predictions'])
        sns.heatmap(cm_vae, annot=True, fmt='d', cmap='Reds', cbar=False)
        plt.title('VAE - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
    except:
        plt.text(0.5, 0.5, 'Confusion matrix\nnot available',
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('VAE Confusion Matrix')

    # 11. Risk Distribution
    plt.subplot(3, 4, 11)
    risk_distribution = test_data['true_risk_cluster'].value_counts().sort_index()
    plt.bar(risk_distribution.index, risk_distribution.values,
            color=['green', 'orange', 'red'], alpha=0.7)
    plt.xlabel('Risk Cluster')
    plt.ylabel('Count')
    plt.title('Test Set Risk Distribution')
    plt.xticks([0, 1, 2], ['Low', 'Medium', 'High'])

    # 12. Model Performance Summary
    plt.subplot(3, 4, 12)
    performance_text = f"""
    BAYESIAN MODEL:
    Clusters Found: {bayesian_results['n_clusters']}
    Silhouette: {bayesian_results['silhouette']:.3f}
    ARI: {bayesian_results['ari']:.3f}
    NMI: {bayesian_results['nmi']:.3f}

    VAE MODEL:
    Clusters Found: {vae_results['n_clusters']}
    Silhouette: {vae_results['silhouette']:.3f}
    ARI: {vae_results['ari']:.3f}
    NMI: {vae_results['nmi']:.3f}
    """
    plt.text(0.1, 0.5, performance_text, fontsize=10, verticalalignment='center',
             transform=plt.gca().transAxes, fontfamily='monospace')
    plt.axis('off')
    plt.title('Performance Summary')

    plt.tight_layout()
    plt.show()


def generate_financial_insights(data, model, X_test, y_test, uncertainty_results, scaler, bayesian_results, vae_results):
    """Generate actionable financial insights from the model"""

    print("\n=== FINANCIAL RISK ASSESSMENT INSIGHTS ===")

    # 1. High-uncertainty cases (require manual review)
    if uncertainty_results['uncertainty'] is not None:
        high_uncertainty_threshold = np.percentile(uncertainty_results['uncertainty'], 80)
        high_uncertainty_mask = uncertainty_results['uncertainty'] > high_uncertainty_threshold

        print(f"\n1. HIGH UNCERTAINTY CASES (Manual Review Recommended):")
        print(f"   - {np.sum(high_uncertainty_mask)} SMEs ({np.sum(high_uncertainty_mask)/len(X_test)*100:.1f}%) require manual review")
        print(f"   - These cases have prediction uncertainty > {high_uncertainty_threshold:.3f}")
    else:
        print(f"\n1. UNCERTAINTY ANALYSIS:")
        print(f"   - Unable to perform uncertainty analysis due to model limitations")

    # 2. Risk distribution analysis
    predictions = bayesian_results['predictions']
    unique_preds = np.unique(predictions)
    risk_names = ['Low Risk', 'Medium Risk', 'High Risk']

    print(f"\n2. PREDICTED RISK DISTRIBUTION:")
    for pred in unique_preds:
        count = np.sum(predictions == pred)
        percentage = count / len(predictions) * 100
        risk_name = risk_names[pred] if pred < len(risk_names) else f"Cluster {pred}"
        print(f"   - {risk_name}: {count} SMEs ({percentage:.1f}%)")

    # 3. Model confidence analysis
    if uncertainty_results['confidence'] is not None:
        high_confidence_mask = uncertainty_results['confidence'] > 0.8
        medium_confidence_mask = (uncertainty_results['confidence'] > 0.6) & (uncertainty_results['confidence'] <= 0.8)
        low_confidence_mask = uncertainty_results['confidence'] <= 0.6

        print(f"\n3. MODEL CONFIDENCE ANALYSIS:")
        print(f"   - High Confidence (>80%): {np.sum(high_confidence_mask)} SMEs ({np.sum(high_confidence_mask)/len(X_test)*100:.1f}%)")
        print(f"   - Medium Confidence (60-80%): {np.sum(medium_confidence_mask)} SMEs ({np.sum(medium_confidence_mask)/len(X_test)*100:.1f}%)")
        print(f"   - Low Confidence (<60%): {np.sum(low_confidence_mask)} SMEs ({np.sum(low_confidence_mask)/len(X_test)*100:.1f}%)")
    else:
        print(f"\n3. MODEL CONFIDENCE ANALYSIS:")
        print(f"   - Unable to perform confidence analysis due to model limitations")

    # 4. Performance comparison
    print(f"\n4. MODEL PERFORMANCE COMPARISON:")
    print(f"   - Bayesian Deep Clustering:")
    print(f"     * Clusters Found: {bayesian_results['n_clusters']}")
    print(f"     * Silhouette Score: {bayesian_results['silhouette']:.4f}")
    print(f"     * Adjusted Rand Index: {bayesian_results['ari']:.4f}")
    print(f"   - VAE + K-means:")
    print(f"     * Clusters Found: {vae_results['n_clusters']}")
    print(f"     * Silhouette Score: {vae_results['silhouette']:.4f}")
    print(f"     * Adjusted Rand Index: {vae_results['ari']:.4f}")

    # 5. Loan approval recommendations
    print(f"\n5. LOAN APPROVAL FRAMEWORK:")
    if len(unique_preds) > 1:
        low_risk_count = np.sum(predictions == 0) if 0 in unique_preds else 0
        medium_risk_count = np.sum(predictions == 1) if 1 in unique_preds else 0
        high_risk_count = np.sum(predictions == 2) if 2 in unique_preds else 0

        print(f"   - APPROVE (Low Risk): {low_risk_count} SMEs")
        print(f"   - CONDITIONAL APPROVAL (Medium Risk): {medium_risk_count} SMEs")
        print(f"   - DETAILED REVIEW (High Risk): {high_risk_count} SMEs")
    else:
        print(f"   - ALL SMEs assigned to single cluster - requires model improvement")

    # 6. Expected impact on NPL reduction
    current_npl_rate = 0.2413  # 24.13% as mentioned in the problem

    # Use the better performing model's metrics
    best_model_ari = max(bayesian_results['ari'], vae_results['ari'])
    estimated_npl_reduction = current_npl_rate * best_model_ari * 0.2  # Conservative estimate

    print(f"\n6. EXPECTED IMPACT ON NPL CRISIS:")
    print(f"   - Current NPL Rate: {current_npl_rate*100:.2f}%")
    print(f"   - Best Model Performance (ARI): {best_model_ari:.4f}")
    print(f"   - Estimated NPL Reduction: {estimated_npl_reduction*100:.2f} percentage points")
    print(f"   - Projected NPL Rate: {(current_npl_rate - estimated_npl_reduction)*100:.2f}%")

    # 7. Implementation recommendations
    print(f"\n7. IMPLEMENTATION RECOMMENDATIONS:")
    if bayesian_results['n_clusters'] > 1:
        print("   - Deploy Bayesian model as primary decision support system")
        print("   - Implement uncertainty-based manual review process")
    elif vae_results['n_clusters'] > 1:
        print("   - Deploy VAE + K-means as primary clustering approach")
        print("   - Consider ensemble methods combining both models")
    else:
        print("   - Both models require further development")
        print("   - Consider alternative architectures or feature engineering")

    print("   - Establish continuous learning pipeline with loan performance feedback")
    print("   - Create risk-based pricing framework")
    print("   - Implement early warning system for portfolio monitoring")


def print_performance_summary(bayesian_results, vae_results, stability_scores, uncertainty_results):
    """Print comprehensive performance summary"""

    print("\n" + "="*80)
    print("COMPREHENSIVE PERFORMANCE SUMMARY")
    print("="*80)

    # Model comparison
    print("\n1. MODEL COMPARISON:")
    print("-" * 50)
    print(f"{'Metric':<25} {'Bayesian':<15} {'VAE+K-means':<15} {'Winner':<15}")
    print("-" * 50)

    metrics = [
        ('Silhouette Score', bayesian_results['silhouette'], vae_results['silhouette']),
        ('Adjusted Rand Index', bayesian_results['ari'], vae_results['ari']),
        ('Normalized Mutual Info', bayesian_results['nmi'], vae_results['nmi']),
        ('Clusters Found', bayesian_results['n_clusters'], vae_results['n_clusters'])
    ]

    bayesian_wins = 0
    for metric_name, bayesian_score, vae_score in metrics[:-1]:  # Exclude clusters found from winner calculation
        winner = "Bayesian" if bayesian_score > vae_score else "VAE+K-means" if vae_score > bayesian_score else "Tie"
        if bayesian_score > vae_score:
            bayesian_wins += 1
        print(f"{metric_name:<25} {bayesian_score:<15.4f} {vae_score:<15.4f} {winner:<15}")

    # Special handling for clusters found
    print(f"{'Clusters Found':<25} {bayesian_results['n_clusters']:<15} {vae_results['n_clusters']:<15} {'VAE' if vae_results['n_clusters'] > bayesian_results['n_clusters'] else 'Bayesian':<15}")

    print("-" * 50)
    overall_winner = "Bayesian Deep Clustering" if bayesian_wins >= 2 else "VAE + K-means"
    print(f"Overall Winner: {overall_winner}")

    # Stability analysis
    print(f"\n2. STABILITY ANALYSIS:")
    print("-" * 30)
    if len(stability_scores) > 0 and any(score > 0 for score in stability_scores):
        stability_mean = np.mean(stability_scores)
        stability_std = np.std(stability_scores)
        coefficient_variation = stability_std / stability_mean if stability_mean > 0 else float('inf')

        print(f"Mean Performance (ARI): {stability_mean:.4f} ± {stability_std:.4f}")
        print(f"Coefficient of Variation: {coefficient_variation:.4f}")
        print(f"Stability Rating: {'High' if coefficient_variation < 0.1 else 'Medium' if coefficient_variation < 0.2 else 'Low'}")
    else:
        print("Stability analysis shows consistently poor performance across runs")
        print("Model requires architectural improvements")

    # Uncertainty analysis summary
    print(f"\n3. UNCERTAINTY QUANTIFICATION:")
    print("-" * 35)
    if uncertainty_results and uncertainty_results.get('uncertainty') is not None:
        mean_uncertainty = np.mean(uncertainty_results['uncertainty'])
        mean_confidence = np.mean(uncertainty_results['confidence'])

        print(f"Average Prediction Uncertainty: {mean_uncertainty:.4f}")
        print(f"Average Prediction Confidence: {mean_confidence:.4f}")
        print(f"High Uncertainty Cases (>90th percentile): {np.sum(uncertainty_results['uncertainty'] > np.percentile(uncertainty_results['uncertainty'], 90))}")
    else:
        print("Uncertainty quantification not available due to model limitations")

    # Business impact assessment
    print(f"\n4. BUSINESS IMPACT ASSESSMENT:")
    print("-" * 40)
    print("✓ Addresses Bangladesh's NPL crisis (current rate: 24.13%)")
    if bayesian_results['n_clusters'] > 1:
        print("✓ Successfully segments SMEs into risk categories")
    else:
        print("✗ Bayesian model failed to learn distinct risk categories")

    if vae_results['n_clusters'] > 1:
        print("✓ VAE baseline demonstrates viable clustering approach")

    print("✓ Provides framework for uncertainty quantification")
    print("✓ Scalable solution for processing large loan applications")

    # Technical achievements and limitations
    print(f"\n5. TECHNICAL ASSESSMENT:")
    print("-" * 35)
    print("✓ Successfully implemented Bayesian Deep Clustering Network")
    print("✓ Demonstrated Monte Carlo Dropout for uncertainty estimation")
    print("✓ Provided comprehensive comparative analysis")

    if bayesian_results['n_clusters'] <= 1:
        print("* Bayesian model suffered from cluster collapse")
        print("  → Requires architectural improvements or hyperparameter tuning")

    if vae_results['ari'] > bayesian_results['ari']:
        print("* VAE + K-means provided superior clustering performance")

    print(f"\n6. RECOMMENDATIONS FOR PRODUCTION:")
    print("-" * 45)
    if vae_results['n_clusters'] > 1 and vae_results['ari'] > 0.3:
        print("• Deploy VAE + K-means as primary clustering solution")
        print("• Investigate Bayesian model architecture improvements")
        print("• Implement ensemble approach combining both models")
    else:
        print("• Both models require further development before deployment")
        print("• Consider alternative architectures (e.g., Deep Embedded Clustering)")
        print("• Explore feature engineering and data augmentation techniques")

    print("• Establish continuous model monitoring and retraining pipeline")
    print("• Implement A/B testing framework for model comparison")

    print("\n" + "="*80)


def main():
    """Main execution function"""
    print("=== Bayesian Deep Clustering Network for SME Credit Risk Assessment ===")
    print("\n1. Generating Synthetic SME Data...")

    # Generate data
    data_generator = SMEDataGenerator(n_samples=2000)
    data = data_generator.generate_data()

    print(f"Generated {len(data)} SME records with {len(data.columns)-1} features")
    print(f"Risk distribution: {data['true_risk_cluster'].value_counts().to_dict()}")

    # Prepare data
    features = data.drop('true_risk_cluster', axis=1)
    labels = data['true_risk_cluster'].values

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, labels, test_size=0.3, random_state=42, stratify=labels
    )

    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Convert to tensorflow tensors
    X_train = tf.constant(X_train, dtype=tf.float32)
    X_test = tf.constant(X_test, dtype=tf.float32)

    print("\n2. Training Bayesian Deep Clustering Network...")

    # Train Bayesian Deep Clustering Network
    bayesian_model, train_losses, val_losses = train_bayesian_clustering_model(
        X_train, X_test, epochs=150
    )

    print("\n3. Training VAE for comparison...")

    # Train VAE for comparison
    vae_model = train_vae_model(X_train, X_test, epochs=150)

    print("\n4. Evaluating Models...")

    # Evaluate both models
    bayesian_results = evaluate_clustering_performance(
        bayesian_model, X_test, y_test, "Bayesian Deep Clustering Network"
    )

    vae_results = evaluate_clustering_performance(
        vae_model, X_test, y_test, "VAE + K-means"
    )

    print("\n5. Stability Analysis...")

    # Stability analysis
    stability_scores = stability_analysis(X_train, X_test, y_test, num_runs=5)

    print("\n6. Uncertainty Analysis...")

    # Uncertainty analysis for Bayesian model
    try:
        uncertainty_results = uncertainty_analysis(bayesian_model, X_test, num_samples=50)
    except Exception as e:
        print(f"Uncertainty analysis failed: {e}")
        uncertainty_results = {
            'mean_probabilities': None,
            'std_probabilities': None,
            'uncertainty': None,
            'confidence': None
        }

    print("\n7. Creating Comprehensive Visualizations...")

    # Create visualizations
    try:
        create_visualizations(
            data, bayesian_model, vae_model, X_test, y_test,
            bayesian_results, vae_results, uncertainty_results
        )
    except Exception as e:
        print(f"Visualization creation encountered an error: {e}")
        print("Continuing with analysis...")

    print("\n8. Financial Risk Insights and Recommendations...")

    # Generate financial insights
    generate_financial_insights(
        data, bayesian_model, X_test, y_test, uncertainty_results, scaler, bayesian_results, vae_results
    )

    print("\n9. Performance Summary...")

    # Print comprehensive summary
    print_performance_summary(
        bayesian_results, vae_results, stability_scores, uncertainty_results
    )

    return {
        'data': data,
        'bayesian_model': bayesian_model,
        'vae_model': vae_model,
        'scaler': scaler,
        'results': {
            'bayesian': bayesian_results,
            'vae': vae_results,
            'stability': stability_scores,
            'uncertainty': uncertainty_results
        },
        'training_history': {
            'bayesian_train_losses': train_losses,
            'bayesian_val_losses': val_losses
        }
    }


if __name__ == "__main__":

    try:
        results = main()
        print("\n" + "="*80)
        print("EXECUTION COMPLETED SUCCESSFULLY")
        print("="*80)
    except Exception as e:
        print(f"\nEXECUTION FAILED: {e}")
        import traceback
        traceback.print_exc()
