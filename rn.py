import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_recall_fscore_support)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Configurações
PASTA_RESULTADOS = "./lista_resultados"
MAX_WORDS = 10000  # Vocabulário máximo
MAX_SEQUENCE_LENGTH = 500  # Tamanho máximo da sequência
EMBEDDING_DIM = 128  # Dimensão do embedding
EPOCHS = 100  # Aumentado para dar mais tempo de convergência
BATCH_SIZE = 16  # Reduzido para melhor aprendizado com poucos dados
VALIDATION_SPLIT = 0.2  # % dos dados de TREINO para validação (ex: 0.2 = 20%)
TEST_SIZE = 0.3  # % de TODOS os dados separados para teste (ex: 0.3 = 30%)
RANDOM_STATE = 42

class SimpleTokenizer:
    """Tokenizador simples para substituir o Keras Tokenizer"""
    
    def __init__(self, num_words=None, oov_token='<OOV>'):
        self.num_words = num_words
        self.oov_token = oov_token
        self.word_index = {}
        self.index_word = {}
        
    def fit_on_texts(self, texts):
        """Cria o vocabulário a partir dos textos"""
        word_counts = {}
        
        for text in texts:
            words = str(text).lower().split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Ordena por frequência
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Cria índices (1-based, 0 reservado para padding)
        self.word_index = {self.oov_token: 1}
        self.index_word = {1: self.oov_token}
        
        idx = 2
        for word, count in sorted_words:
            if self.num_words and idx > self.num_words:
                break
            self.word_index[word] = idx
            self.index_word[idx] = word
            idx += 1
    
    def texts_to_sequences(self, texts):
        """Converte textos em sequências de inteiros"""
        sequences = []
        oov_idx = self.word_index.get(self.oov_token, 1)
        
        for text in texts:
            words = str(text).lower().split()
            seq = []
            for word in words:
                idx = self.word_index.get(word, oov_idx)
                seq.append(idx)
            sequences.append(seq)
        
        return sequences

def pad_sequences(sequences, maxlen=None, padding='post', truncating='post', value=0):
    """Preenche sequências para o mesmo tamanho"""
    if maxlen is None:
        maxlen = max(len(seq) for seq in sequences)
    
    padded = np.zeros((len(sequences), maxlen), dtype=np.int32)
    
    for i, seq in enumerate(sequences):
        if len(seq) == 0:
            continue
            
        if truncating == 'pre':
            trunc = seq[-maxlen:]
        else:  # 'post'
            trunc = seq[:maxlen]
        
        if padding == 'post':
            padded[i, :len(trunc)] = trunc
        else:  # 'pre'
            padded[i, -len(trunc):] = trunc
    
    return padded

def carregar_todos_csvs(pasta):
    """
    Carrega todos os arquivos CSV de notícias processadas
    """
    print("=== CARREGANDO DADOS ===")
    print(f"Pasta alvo: {pasta}")
    todos_dataframes = []
    
    # Percorre todas as subpastas em lista_resultados
    pasta_path = Path(pasta)
    
    print(f"Verificando se pasta existe...")
    if not pasta_path.exists():
        print(f"✗ ERRO: Pasta {pasta} não encontrada!")
        print(f"   Caminho absoluto: {pasta_path.absolute()}")
        return None
    
    print(f"✓ Pasta encontrada!")
    print(f"Procurando arquivos CSV recursivamente...")
    
    arquivos_csv = list(pasta_path.rglob("noticias_g1_processadas.csv"))
    
    print(f"Arquivos encontrados: {len(arquivos_csv)}")
    
    if not arquivos_csv:
        print(f"✗ Nenhum arquivo CSV encontrado em {pasta}")
        print(f"\nDica: Certifique-se que você executou o scraper (main.py) antes!")
        print(f"      Os arquivos devem estar em subpastas como:")
        print(f"      {pasta}/YYYYMMDD_HHMMSS/noticias_g1_processadas.csv")
        return None
    
    print(f"\n✓ Encontrados {len(arquivos_csv)} arquivo(s) CSV:")
    
    for arquivo in arquivos_csv:
        try:
            print(f"\n  Carregando: {arquivo.name}")
            print(f"    Caminho: {arquivo}")
            df = pd.read_csv(arquivo, sep=';', encoding='utf-8-sig')
            print(f"    ✓ {len(df)} notícias carregadas")
            todos_dataframes.append(df)
        except Exception as e:
            print(f"    ✗ Erro: {e}")
    
    if todos_dataframes:
        print(f"\nConcatenando dataframes...")
        df_final = pd.concat(todos_dataframes, ignore_index=True)
        print(f"✓ Total de notícias carregadas: {len(df_final)}")
        print(f"  Colunas: {list(df_final.columns)}")
        return df_final
    else:
        print("✗ Nenhum dataframe foi carregado com sucesso")
        return None

def preprocessar_dados(df, balancear=False):
    """
    Preprocessa os dados para o modelo
    """
    print("\n=== PREPROCESSAMENTO ===")
    
    # Remove linhas com valores nulos em campos essenciais
    df = df.dropna(subset=['texto', 'titulo', 'genero'])
    
    # Remove notícias sem texto
    df = df[df['texto'].str.strip() != '']
    
    # Remove gêneros com poucas amostras (menos de 2)
    contagem_generos = df['genero'].value_counts()
    print(f"\nDistribuição original por gênero:")
    print(contagem_generos)
    
    generos_validos = contagem_generos[contagem_generos >= 2].index
    df = df[df['genero'].isin(generos_validos)]
    
    print(f"\nNotícias após limpeza: {len(df)}")
    print(f"\nDistribuição final por gênero:")
    distribuicao_final = df['genero'].value_counts()
    print(distribuicao_final)
    
    # Alerta sobre desbalanceamento
    max_count = distribuicao_final.max()
    min_count = distribuicao_final.min()
    ratio = max_count / min_count if min_count > 0 else float('inf')
    
    if ratio > 3:
        print(f"\n⚠️  AVISO: Dataset MUITO desbalanceado!")
        print(f"   Razão máx/mín: {ratio:.1f}x")
        print(f"   Isso DEFINITIVAMENTE fará o modelo favorecer classes majoritárias")
        print(f"   Soluções aplicadas:")
        print(f"   1. class_weight para penalizar erros nas classes minoritárias")
        print(f"   2. Oversampling manual das classes minoritárias")
        
        if balancear and ratio > 5:
            print(f"\n🔄 Aplicando oversampling nas classes minoritárias...")
            dfs_balanceados = []
            target_count = int(max_count * 0.7)  # 70% da classe majoritária
            
            for genero in df['genero'].unique():
                df_genero = df[df['genero'] == genero]
                count = len(df_genero)
                
                if count < target_count:
                    # Replica amostras para aumentar a classe
                    n_replicas = (target_count // count) + 1
                    df_replicado = pd.concat([df_genero] * n_replicas, ignore_index=True)
                    df_replicado = df_replicado.sample(n=target_count, random_state=42)
                    dfs_balanceados.append(df_replicado)
                    print(f"   {genero}: {count} → {target_count} amostras")
                else:
                    dfs_balanceados.append(df_genero)
                    print(f"   {genero}: {count} amostras (mantido)")
            
            df = pd.concat(dfs_balanceados, ignore_index=True)
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
            
            print(f"\n✓ Dataset balanceado! Nova distribuição:")
            print(df['genero'].value_counts())
    
    return df

def criar_modelo_rnn(vocab_size, num_classes, max_length):
    """
    Cria uma Rede Neural otimizada para datasets pequenos
    """
    model = keras.Sequential([
        # Camada de Embedding
        layers.Embedding(input_dim=vocab_size, 
                        output_dim=EMBEDDING_DIM, 
                        input_length=max_length,
                        mask_zero=True),  # Ignora padding
        
        # Dropout mais leve
        layers.Dropout(0.2),
        
        # LSTM bidirecional menor para evitar overfitting
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Dropout(0.2),
        
        # Segunda camada LSTM
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dropout(0.2),
        
        # Camadas densas menores
        layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.Dropout(0.2),
        
        # Camada de saída
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def plotar_metricas_treinamento(history, pasta_saida):
    """
    Plota gráficos de accuracy e loss durante o treinamento
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Treino', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validação', linewidth=2)
    axes[0].set_title('Acurácia do Modelo', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Acurácia')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Treino', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validação', linewidth=2)
    axes[1].set_title('Loss do Modelo', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{pasta_saida}/metricas_treinamento.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Gráficos salvos em: {pasta_saida}/metricas_treinamento.png")
    plt.close()

def plotar_matriz_confusao(y_true, y_pred, classes, pasta_saida):
    """
    Plota a matriz de confusão com TODAS as classes
    """
    # Criar matriz de confusão com todas as classes
    all_labels = np.arange(len(classes))
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    
    # Ajustar tamanho da figura baseado no número de classes
    fig_size = max(10, len(classes) * 1.5)
    plt.figure(figsize=(fig_size, fig_size))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Quantidade'})
    plt.title('Matriz de Confusão - Todos os Gêneros', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Classe Real', fontsize=12)
    plt.xlabel('Classe Predita', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{pasta_saida}/matriz_confusao.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Matriz de confusão salva em: {pasta_saida}/matriz_confusao.png")
    plt.close()

def plotar_metricas_por_classe(report_dict, classes, pasta_saida):
    """
    Plota métricas (precision, recall, f1-score) por classe - TODAS as classes
    """
    metricas = ['precision', 'recall', 'f1-score']
    valores = {metrica: [] for metrica in metricas}
    
    for classe in classes:
        for metrica in metricas:
            # Pega o valor ou 0 se a classe não tiver dados
            valores[metrica].append(report_dict.get(classe, {}).get(metrica, 0))
    
    x = np.arange(len(classes))
    width = 0.25
    
    # Ajustar tamanho da figura baseado no número de classes
    fig_width = max(14, len(classes) * 2)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    
    rects1 = ax.bar(x - width, valores['precision'], width, label='Precision', alpha=0.8)
    rects2 = ax.bar(x, valores['recall'], width, label='Recall', alpha=0.8)
    rects3 = ax.bar(x + width, valores['f1-score'], width, label='F1-Score', alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Métricas por Gênero de Notícia - Todos os Gêneros', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    # Adiciona valores nas barras
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:  # Só mostra se tiver valor
                ax.annotate(f'{height:.2f}',
                           xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.tight_layout()
    plt.savefig(f'{pasta_saida}/metricas_por_classe.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Métricas por classe salvas em: {pasta_saida}/metricas_por_classe.png")
    plt.close()

def main():
    print("\n" + "="*60)
    print("  CLASSIFICADOR DE NOTÍCIAS G1 COM REDE NEURAL LSTM")
    print("="*60)
    print(f"Iniciando em: {pd.Timestamp.now()}")
    print("="*60 + "\n")
    
    # Criar pasta para resultados
    pasta_saida = "./resultados_rn"
    print(f"Criando pasta de saída: {pasta_saida}")
    Path(pasta_saida).mkdir(exist_ok=True)
    print("✓ Pasta criada\n")
    
    # 1. Carregar dados
    print(f"Procurando CSVs em: {PASTA_RESULTADOS}")
    df = carregar_todos_csvs(PASTA_RESULTADOS)
    
    if df is None or len(df) == 0:
        print("\n✗ Nenhum dado foi carregado. Encerrando.")
        return
    
    # 2. Preprocessar (com balanceamento se necessário)
    df = preprocessar_dados(df, balancear=True)  # ATIVA O OVERSAMPLING
    
    if len(df) < 2:
        print("\n✗ Dados insuficientes para treinamento. Mínimo: 2 notícias")
        return
    
    # 3. Preparar dados para o modelo
    print("\n=== PREPARANDO DADOS PARA O MODELO ===")
    
    # OPÇÃO 1: Usar tokens lemmatizados (mais processados) do CSV
    print("Verificando se existem tokens processados no CSV...")
    
    usar_tokens_csv = False
    if 'tokens_lemmatizados' in df.columns:
        print("✓ Encontrados tokens lemmatizados! Usando esses ao invés de reprocessar.")
        # Converter string de lista para lista real
        try:
            import ast
            textos = df['tokens_lemmatizados'].apply(lambda x: ' '.join(ast.literal_eval(x)) if pd.notna(x) else '').values
            usar_tokens_csv = True
            print("✓ Tokens lemmatizados carregados com sucesso")
        except:
            print("⚠️  Erro ao processar tokens_lemmatizados, usando texto original")
            textos = df['titulo'].astype(str) + " " + df['texto'].astype(str)
    elif 'tokens_stemmed' in df.columns:
        print("✓ Encontrados tokens stemmed! Usando esses ao invés de reprocessar.")
        try:
            import ast
            textos = df['tokens_stemmed'].apply(lambda x: ' '.join(ast.literal_eval(x)) if pd.notna(x) else '').values
            usar_tokens_csv = True
            print("✓ Tokens stemmed carregados com sucesso")
        except:
            print("⚠️  Erro ao processar tokens_stemmed, usando texto original")
            textos = df['titulo'].astype(str) + " " + df['texto'].astype(str)
    else:
        print("⚠️  Tokens processados não encontrados, usando texto original")
        textos = df['titulo'].astype(str) + " " + df['texto'].astype(str)
    
    labels = df['genero'].values
    
    if usar_tokens_csv:
        print(f"📊 Exemplo de tokens processados (primeira notícia):")
        print(f"   {textos[0][:200]}...")
    else:
        print(f"📊 Exemplo de texto original (primeira notícia):")
        print(f"   {textos[0][:200]}...")
    
    # Codificar labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    
    print(f"Número de classes: {num_classes}")
    print(f"Classes: {list(label_encoder.classes_)}")
    
    # Calcular pesos das classes para lidar com desbalanceamento
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_encoded),
        y=y_encoded
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    print(f"\n⚙️  Pesos das classes (para balanceamento):")
    for idx, classe in enumerate(label_encoder.classes_):
        print(f"   {classe}: {class_weight_dict[idx]:.2f}")
    
    # Tokenização usando nossa classe customizada
    print("\nTokenizando textos...")
    tokenizer = SimpleTokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
    tokenizer.fit_on_texts(textos)
    sequences = tokenizer.texts_to_sequences(textos)
    X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    
    vocab_size = min(len(tokenizer.word_index) + 1, MAX_WORDS)
    print(f"Tamanho do vocabulário: {vocab_size}")
    
    # Dividir em treino e teste com estratificação
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
        )
    except ValueError:
        # Se houver classe com apenas 1 amostra, não usa estratificação
        print("⚠️  Aviso: Algumas classes têm poucas amostras. Dividindo sem estratificação.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
    
    print(f"\nConjunto de treino: {len(X_train)} amostras")
    print(f"Conjunto de teste: {len(X_test)} amostras")
    
    # Mostrar distribuição por classe
    print(f"\nDistribuição no conjunto de treino:")
    unique, counts = np.unique(y_train, return_counts=True)
    for idx, count in zip(unique, counts):
        print(f"   {label_encoder.classes_[idx]}: {count}")
    
    # 4. Criar e compilar modelo
    print("\n=== CRIANDO MODELO ===")
    model = criar_modelo_rnn(vocab_size, num_classes, MAX_SEQUENCE_LENGTH)
    
    # Usar learning rate menor para melhor convergência
    optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nArquitetura do modelo:")
    model.summary()
    
    # 5. Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,  # Mais paciência
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1
    )
    
    # 6. Treinar modelo COM class_weight
    print("\n=== TREINANDO MODELO ===")
    print(f"Epochs: {EPOCHS} | Batch size: {BATCH_SIZE}")
    print(f"Usando class_weight para compensar desbalanceamento")
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        class_weight=class_weight_dict,  # IMPORTANTE: Compensa desbalanceamento
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # 7. Avaliar modelo
    print("\n=== AVALIANDO MODELO ===")
    
    # Predições
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Mostrar distribuição das predições
    print(f"\nDistribuição das predições:")
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    for idx, count in zip(unique_pred, counts_pred):
        print(f"   {label_encoder.classes_[idx]}: {count}")
    
    # Métricas gerais
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✓ Acurácia no conjunto de teste: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Relatório de classificação
    print("\n" + "="*60)
    print("RELATÓRIO DE CLASSIFICAÇÃO")
    print("="*60)
    
    # Usar TODAS as classes do dataset para o relatório
    all_classes = np.arange(len(label_encoder.classes_))
    
    report = classification_report(
        y_test, y_pred, 
        labels=all_classes,
        target_names=label_encoder.classes_,
        digits=4,
        zero_division=0
    )
    print(report)
    
    # Salvar relatório em arquivo
    with open(f'{pasta_saida}/relatorio_classificacao.txt', 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("RELATÓRIO DE CLASSIFICAÇÃO - NOTÍCIAS G1\n")
        f.write("="*60 + "\n\n")
        f.write(f"Acurácia: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        f.write(f"Distribuição das predições:\n")
        for idx, count in zip(unique_pred, counts_pred):
            f.write(f"   {label_encoder.classes_[idx]}: {count}\n")
        f.write("\n")
        f.write(report)
    
    print(f"\n✓ Relatório salvo em: {pasta_saida}/relatorio_classificacao.txt")
    
    # 8. Gerar visualizações
    print("\n=== GERANDO VISUALIZAÇÕES ===")
    
    # Gráficos de treinamento
    plotar_metricas_treinamento(history, pasta_saida)
    
    # Matriz de confusão - usar TODAS as classes
    plotar_matriz_confusao(y_test, y_pred, label_encoder.classes_, pasta_saida)
    
    # Métricas por classe - usar TODAS as classes
    report_dict = classification_report(
        y_test, y_pred,
        labels=all_classes,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )
    plotar_metricas_por_classe(report_dict, label_encoder.classes_, pasta_saida)
    
    # 9. Salvar modelo
    print("\n=== SALVANDO MODELO ===")
    model.save(f'{pasta_saida}/modelo_classificador_g1.keras')
    print(f"✓ Modelo salvo em: {pasta_saida}/modelo_classificador_g1.keras")
    
    # Salvar tokenizer e label encoder
    import pickle
    with open(f'{pasta_saida}/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    with open(f'{pasta_saida}/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"✓ Tokenizer e Label Encoder salvos")

if __name__ == "__main__":
    main()