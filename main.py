import os
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import re
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import pandas as pd
import spacy

# Baixar recursos necessários do NLTK (executar apenas na primeira vez)
try:
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/rslp')
except LookupError:
    print("Baixando recursos do NLTK...")
    nltk.download('punkt_tab')
    time.sleep(0.5)
    nltk.download('punkt')
    time.sleep(0.5)
    nltk.download('stopwords')
    time.sleep(0.5)
    nltk.download('rslp')

# Carregar modelo do spaCy para lematização
try:
    nlp = spacy.load('pt_core_news_sm')
    print("Modelo spaCy carregado com sucesso!")
except OSError:
    print("AVISO: Modelo spaCy não encontrado. Execute: python -m spacy download pt_core_news_sm")
    nlp = None

def acessar_g1():
    chrome_options = Options()
   
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')  # Evita detecção de bot
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
   
    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        # Maximiza a janela do navegador (tela cheia)
        driver.maximize_window()
        
        driver.get("https://g1.globo.com")
       
        # espera carregar
        wait = WebDriverWait(driver, 10)
        # espera o título ser G1
        wait.until(EC.title_contains("g1 - O portal de notícias da Globo"))
        return driver
       
    except Exception as e:
        print(f"Erro ao acessar o site: {str(e)}")
        if 'driver' in locals():
            driver.quit()
        return None

def percorreNoticias(driver):
    links_noticias = []
    
    try:
        print("Iniciando coleta de links de notícias...")
        
        # 1. Fazer scroll até não conseguir mais
        print("Fazendo scroll na página...")
        ultima_altura = driver.execute_script("return document.body.scrollHeight")
        
        while True:
            # Scroll para baixo
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            
            # Aguarda carregar novo conteúdo
            time.sleep(2)
            
            # Verifica se a altura da página mudou
            nova_altura = driver.execute_script("return document.body.scrollHeight")
            if nova_altura == ultima_altura:
                print("Não há mais conteúdo para carregar")
                break
            ultima_altura = nova_altura
        
        # 2. Primeiro busca todas as div.bstn-hl-wrapper e salva os hrefs
        print("Buscando div.bstn-hl-wrapper...")
        wrappers = driver.find_elements(By.CSS_SELECTOR, "div.bstn-hl-wrapper")
        for wrapper in wrappers:
            try:
                link = wrapper.find_element(By.TAG_NAME, "a")
                href = link.get_attribute("href")
                if href and href not in links_noticias:
                    links_noticias.append(href)
            except:
                continue
        
        print(f"Encontrados {len(links_noticias)} links em bstn-hl-wrapper")
        
        # 3. Busca todos os div.bastian-page com div._evg dentro
        print("Buscando div.bastian-page...")
        bastian_pages = driver.find_elements(By.CSS_SELECTOR, "div.bastian-page")
        
        for page in bastian_pages:
            try:
                # Verifica se tem div._evg dentro
                evg_div = page.find_element(By.CSS_SELECTOR, "div._evg")
                
                # Busca todos os div._evt dentro de _evg
                evt_divs = evg_div.find_elements(By.CSS_SELECTOR, "div._evt")
                
                for evt in evt_divs:
                    try:
                        # Verifica se tem div.post-minicapa-v2__content
                        minicapa = evt.find_elements(By.CSS_SELECTOR, "div.post-minicapa-v2__content")
                        
                        if minicapa:
                            # Caso tenha minicapa - busca articles dentro
                            print("Encontrado div.post-minicapa-v2__content - buscando articles...")
                            for mini in minicapa:
                                articles = mini.find_elements(By.TAG_NAME, "article")
                                for article in articles:
                                    try:
                                        link = article.find_element(By.TAG_NAME, "a")
                                        href = link.get_attribute("href")
                                        if href and href not in links_noticias:
                                            links_noticias.append(href)
                                    except:
                                        continue
                        else:
                            # Caso não tenha minicapa - aplicar lógica das classes específicas
                            try:
                                # Busca links com classes específicas feed-post-link gui-color-primary gui-color-hover
                                links_principais = evt.find_elements(By.CSS_SELECTOR, "a.feed-post-link.gui-color-primary.gui-color-hover")
                                for link in links_principais:
                                    href = link.get_attribute("href")
                                    if href and href not in links_noticias:
                                        links_noticias.append(href)
                                
                                # Verifica se tem ul dentro do evt
                                uls = evt.find_elements(By.TAG_NAME, "ul")
                                for ul in uls:
                                    # Percorre cada li dentro do ul
                                    lis = ul.find_elements(By.TAG_NAME, "li")
                                    for li in lis:
                                        # Busca links com as classes específicas dentro do li
                                        links_relacionados = li.find_elements(By.CSS_SELECTOR, "a.gui-color-primary.gui-color-hover.feed-post-body-title.bstn-relatedtext")
                                        for link in links_relacionados:
                                            href = link.get_attribute("href")
                                            if href and href not in links_noticias:
                                                links_noticias.append(href)
                                                
                            except:
                                continue
                                
                    except:
                        continue
                        
            except:
                continue
        
        print(f"Coleta finalizada! Total de {len(links_noticias)} links únicos encontrados")
        
        for i, link in enumerate(links_noticias):
            print(f"{i+1}. {link}")
        
        return links_noticias
        
    except Exception as e:
        print(f"Erro durante a coleta de notícias: {str(e)}")
        return links_noticias

def processar_texto_completo(texto):
    """
    Processa o texto seguindo a ordem específica:
    1. Tokenização
    2. Normalização  
    3. Remoção de ruídos
    4. Remoção de stopwords e pontuação
    5. Stemming
    6. Lematização
    """
    try:
        print("    → Iniciando processamento de texto...")
        resultado = {}
        
        # PASSO 1: TOKENIZAÇÃO
        print("    → Passo 1: Tokenização")
        sentencas_originais = sent_tokenize(texto, language='portuguese')
        tokens_originais = word_tokenize(texto, language='portuguese')
        
        resultado['sentencas_originais'] = sentencas_originais
        resultado['total_sentencas'] = len(sentencas_originais)
        resultado['tokens_originais'] = tokens_originais
        resultado['total_tokens_originais'] = len(tokens_originais)
        
        # PASSO 2: NORMALIZAÇÃO
        print("    → Passo 2: Normalização")
        tokens_normalizados = []
        for token in tokens_originais:
            # Converte para minúsculas
            token_norm = token.lower()
            # Remove espaços em branco extras
            token_norm = token_norm.strip()
            tokens_normalizados.append(token_norm)
        
        resultado['tokens_normalizados'] = tokens_normalizados
        
        # PASSO 3: REMOÇÃO DE RUÍDOS (tags sem uso)
        print("    → Passo 3: Remoção de ruídos")
        tokens_sem_ruido = []
        for token in tokens_normalizados:
            # Remove tokens vazios ou muito pequenos
            if len(token) < 2:
                continue
            # Remove números puros
            if token.isdigit():
                continue
            # Remove tokens que são apenas símbolos especiais
            if not re.search(r'[a-záàâãçéêíóôõú]', token):
                continue
            # Remove URLs, emails, hashtags, etc.
            if any(pattern in token for pattern in ['http', 'www', '@', '#', '.com', '.br']):
                continue
            
            tokens_sem_ruido.append(token)
        
        resultado['tokens_sem_ruido'] = tokens_sem_ruido
        resultado['total_tokens_sem_ruido'] = len(tokens_sem_ruido)
        
        # PASSO 4: REMOÇÃO DE STOPWORDS E PONTUAÇÃO
        print("    → Passo 4: Remoção de stopwords e pontuação")
        stop_words = set(stopwords.words('portuguese'))
        # Adiciona stopwords customizadas comuns em notícias
        stop_words.update(['disse', 'diz', 'afirmou', 'segundo', 'conforme', 'informou', 
                          'explicou', 'contou', 'relatou', 'declarou', 'comentou'])
        
        tokens_sem_stopwords = []
        for token in tokens_sem_ruido:
            # Remove pontuação
            if not token.isalpha():
                continue
            # Remove stopwords
            if token in stop_words:
                continue
            # Mantém apenas palavras com pelo menos 3 caracteres
            if len(token) >= 3:
                tokens_sem_stopwords.append(token)
        
        resultado['tokens_sem_stopwords'] = tokens_sem_stopwords
        resultado['total_tokens_sem_stopwords'] = len(tokens_sem_stopwords)
        
        # PASSO 5: STEMMING
        print("    → Passo 5: Stemming")
        stemmer = RSLPStemmer()
        tokens_stemmed = []
        for token in tokens_sem_stopwords:
            stem = stemmer.stem(token)
            tokens_stemmed.append(stem)
        
        resultado['tokens_stemmed'] = tokens_stemmed
        resultado['total_tokens_stemmed'] = len(tokens_stemmed)
        resultado['stems_unicos'] = list(set(tokens_stemmed))
        resultado['total_stems_unicos'] = len(set(tokens_stemmed))
        
        # PASSO 6: LEMATIZAÇÃO
        print("    → Passo 6: Lematização")
        if nlp is not None:
            # Reconstrói o texto limpo para processar com spaCy
            texto_limpo = ' '.join(tokens_sem_stopwords)
            doc = nlp(texto_limpo)
            
            tokens_lemmatizados = []
            lemmas_info = []
            
            for token in doc:
                if not token.is_stop and token.is_alpha and len(token.lemma_) >= 3:
                    tokens_lemmatizados.append(token.lemma_.lower())
                    lemmas_info.append({
                        'palavra': token.text,
                        'lemma': token.lemma_.lower(),
                        'pos': token.pos_,
                        'tag': token.tag_
                    })
            
            resultado['tokens_lemmatizados'] = tokens_lemmatizados
            resultado['total_tokens_lemmatizados'] = len(tokens_lemmatizados)
            resultado['lemmas_unicos'] = list(set(tokens_lemmatizados))
            resultado['total_lemmas_unicos'] = len(set(tokens_lemmatizados))
            resultado['lemmas_info'] = lemmas_info
        else:
            # Fallback se spaCy não estiver disponível
            resultado['tokens_lemmatizados'] = tokens_sem_stopwords
            resultado['total_tokens_lemmatizados'] = len(tokens_sem_stopwords)
            resultado['lemmas_unicos'] = list(set(tokens_sem_stopwords))
            resultado['total_lemmas_unicos'] = len(set(tokens_sem_stopwords))
            resultado['lemmas_info'] = []
        
        # MÉTRICAS ADICIONAIS
        resultado['densidade_lexical'] = round(
            resultado['total_lemmas_unicos'] / max(resultado['total_tokens_originais'], 1), 4)
        resultado['reducao_stemming'] = round(
            (resultado['total_tokens_sem_stopwords'] - resultado['total_stems_unicos']) / 
            max(resultado['total_tokens_sem_stopwords'], 1), 4)
        resultado['reducao_lematizacao'] = round(
            (resultado['total_tokens_sem_stopwords'] - resultado['total_lemmas_unicos']) / 
            max(resultado['total_tokens_sem_stopwords'], 1), 4)
        
        print(f"    ✓ Processamento concluído:")
        print(f"      - Tokens originais: {resultado['total_tokens_originais']}")
        print(f"      - Após limpeza: {resultado['total_tokens_sem_stopwords']}")
        print(f"      - Stems únicos: {resultado['total_stems_unicos']}")
        print(f"      - Lemmas únicos: {resultado['total_lemmas_unicos']}")
        
        return resultado
        
    except Exception as e:
        print(f"     Erro no processamento: {e}")
        return {
            'total_sentencas': 0,
            'total_tokens_originais': 0,
            'total_tokens_sem_ruido': 0,
            'total_tokens_sem_stopwords': 0,
            'total_tokens_stemmed': 0,
            'total_stems_unicos': 0,
            'total_tokens_lemmatizados': 0,
            'total_lemmas_unicos': 0,
            'densidade_lexical': 0,
            'reducao_stemming': 0,
            'reducao_lematizacao': 0,
            'lemmas_unicos': [],
            'stems_unicos': []
        }

def salvar_lista(lista, nome_arquivo):
    """Salva uma lista (tokens, stems, lemmas etc.) em arquivo texto."""
    try:
        with open(nome_arquivo, "w", encoding="utf-8") as f:
            for item in lista:
                f.write(str(item) + "\n")
        print(f"     Lista salva em {nome_arquivo}")
    except Exception as e:
        print(f"     Erro ao salvar {nome_arquivo}: {e}")


def extrairDadosNoticias(driver, links):
    """
    Extrai dados detalhados de cada notícia dos links fornecidos
    """
    # links = links[:4]  # REMOVER DPS, só pra agilizar agora
    noticias = []
    
    print(f"Iniciando extração de dados de {len(links)} notícias...")
    
    for i, link in enumerate(links):
        try:
            if link.find('g1.globo.com') == -1 or "/ao-vivo/" in link:
                continue 
            
            print(f"\nProcessando notícia {i+1}/{len(links)}: {link}")
            
            driver.get(link)
            time.sleep(2)
            
            noticia = {
                'genero': '',
                'titulo': '',
                'subtitulo': '',
                'autor': '',
                'time': '',
                'texto': '',
                'link': link,
                'data_coleta': datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            }

            # Extrai o gênero
            try:
                genero_elem = driver.find_element(By.CLASS_NAME, "header-editoria--link")
                noticia['genero'] = genero_elem.text.strip()
            except:
                print(f"  Título não encontrado para: {link}")

            # Busca a div.mc-article-header para extrair título, subtítulo, autor e time
            try:
                header = driver.find_element(By.CSS_SELECTOR, "div.mc-article-header")
                
                
                # Extrai o título (h1)
                try:
                    titulo_elem = header.find_element(By.TAG_NAME, "h1")
                    noticia['titulo'] = titulo_elem.text.strip()
                except:
                    print(f"  Título não encontrado para: {link}")
                
                # Extrai o subtítulo (h2)
                try:
                    subtitulo_elem = header.find_element(By.TAG_NAME, "h2")
                    noticia['subtitulo'] = subtitulo_elem.text.strip()
                except:
                    print(f"  Subtítulo não encontrado para: {link}")
                
                # Extrai o autor (a.multi_signatures)
                try:
                    autor_elem = header.find_element(By.CSS_SELECTOR, "a.multi_signatures")
                    noticia['autor'] = autor_elem.text.strip()
                except:
                    print(f"  Autor não encontrado para: {link}")
                
                # Extrai o time (elemento time)
                try:
                    time_elem = header.find_element(By.TAG_NAME, "time")
                    noticia['time'] = time_elem.text.strip()
                except:
                    print(f"  Time não encontrado para: {link}")
                    
            except:
                print(f"  Header não encontrado para: {link}")
                continue
            
            # Fazer scroll até o final da página para carregar todo o conteúdo
            print(f"  Fazendo scroll para carregar todo o conteúdo...")
            ultima_altura = driver.execute_script("return document.body.scrollHeight")
            
            while True:
                # Scroll para baixo
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                
                # Aguarda carregar novo conteúdo
                time.sleep(1)
                
                # Verifica se a altura da página mudou
                nova_altura = driver.execute_script("return document.body.scrollHeight")
                if nova_altura == ultima_altura:
                    break
                ultima_altura = nova_altura
            
            # Extrai texto dos parágrafos e blockquotes na ordem que aparecem
            try:
                # Busca todos os elementos na ordem que aparecem na página
                # Extrair corpo do texto
                elementos = driver.find_elements(By.CSS_SELECTOR, "p.content-text__container, blockquote.content-blockquote")
                texto_completo = []
                for elemento in elementos:
                    if elemento.tag_name == 'p':
                        # Para parágrafos, extrai apenas texto direto (não de elementos filhos como 'a')
                        texto_p = driver.execute_script("""
                            var element = arguments[0];
                            var text = '';
                            for (var i = 0; i < element.childNodes.length; i++) {
                                var node = element.childNodes[i];
                                text += node.textContent.replaceAll(';',' ').replaceAll(',', ' ');
                            }
                            return text.trim();
                        """, elemento)
                        
                        if texto_p:
                            texto_completo.append(texto_p)
                    
                    elif elemento.tag_name == 'blockquote':
                        # Para blockquotes, pega todo o texto
                        texto = elemento.text.strip()
                        if texto:
                            texto_completo.append(texto)

                noticia['texto'] = ' '.join(texto_completo)

                # Processar texto
                texto_completo_para_processar = f"{noticia['titulo']} {noticia['subtitulo']} {noticia['texto']}"
                if texto_completo_para_processar.strip():
                    resultado_processamento = processar_texto_completo(texto_completo_para_processar)
                    noticia.update({
                        'tokens_originais': resultado_processamento['tokens_originais'],
                        'tokens_normalizados': resultado_processamento['tokens_normalizados'],
                        'tokens_stemmed': resultado_processamento['tokens_stemmed'],
                        'tokens_lemmatizados': resultado_processamento['tokens_lemmatizados'],
                        'total_sentencas': resultado_processamento['total_sentencas'],
                        'total_tokens_originais': resultado_processamento['total_tokens_originais'],
                        'total_tokens_sem_ruido': resultado_processamento['total_tokens_sem_ruido'],
                        'total_tokens_sem_stopwords': resultado_processamento['total_tokens_sem_stopwords'],
                        'total_stems_unicos': resultado_processamento['total_stems_unicos'],
                        'total_lemmas_unicos': resultado_processamento['total_lemmas_unicos'],
                        'densidade_lexical': resultado_processamento['densidade_lexical'],
                    })

            except Exception as e:
                print(f"  Erro ao extrair texto para: {link} - {e}")

            noticias.append(noticia)
            print(f"   Notícia processada e arquivos salvos!")
                
        except Exception as e:
            print(f"   Erro ao processar {link}: {e}")
            continue
    
    return noticias

def salvar_csv(noticias, cDir):
    """
    Salva as notícias em arquivo CSV
    """
    if not noticias:
        print("Nenhuma notícia para salvar.")
        return
    
    nome_arquivo = f'{cDir}noticias_g1_processadas.csv'
    
    try:
        # Cria um DataFrame do pandas
        df = pd.DataFrame(noticias)
        
        # Salva em CSV
        df.to_csv(nome_arquivo, index=False, encoding='utf-8-sig', sep=';')
        
        print(f"\n Dados salvos em: {nome_arquivo}")
        print(f"Total de notícias salvas: {len(noticias)}")
        
        # Mostra estatísticas gerais
        if 'total_tokens_originais' in df.columns:
            print(f"\n=== ESTATÍSTICAS DO PROCESSAMENTO ===")
            print(f"Média de tokens originais por notícia: {df['total_tokens_originais'].mean():.1f}")
            print(f"Média de sentenças por notícia: {df['total_sentencas'].mean():.1f}")
            print(f"Média de tokens após limpeza: {df['total_tokens_sem_stopwords'].mean():.1f}")
            print(f"Média de stems únicos: {df['total_stems_unicos'].mean():.1f}")
            print(f"Média de lemmas únicos: {df['total_lemmas_unicos'].mean():.1f}")
            print(f"Densidade lexical média: {df['densidade_lexical'].mean():.4f}")
            print(f"Redução média por stemming: {df['reducao_stemming'].mean():.2%}")
            print(f"Redução média por lematização: {df['reducao_lematizacao'].mean():.2%}")
            print(f"Total de tokens coletados: {df['total_tokens_originais'].sum()}")
        
        return nome_arquivo
        
    except Exception as e:
        print(f"Erro ao salvar CSV: {e}")
        return None

def main():
    print("=== SCRAPER G1 COM PROCESSAMENTO LINGUÍSTICO COMPLETO ===")
    print("Pipeline: Tokenização → Normalização → Remoção de Ruídos → Stopwords → Stemming → Lematização\n")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cDirRun = Path("./lista_resultados") / timestamp
    cDirRun.mkdir(parents=True, exist_ok=True)

    cDirRun = "./lista_resultados/"+timestamp+"/"
        
    # Acessar o G1
    driver = acessar_g1()
   
    if driver:
        print("Acesso ao G1 realizado com sucesso!")
        
        # Coleta os links das notícias
        links = percorreNoticias(driver)
        
        if links:
            print(f"\n=== INICIANDO EXTRAÇÃO E PROCESSAMENTO ===")
            # Extrai os dados detalhados de cada notícia
            noticias = extrairDadosNoticias(driver, links)
            
            tokens_originais = []
            tokens_normalizados = []
            tokens_stemmed = []
            tokens_lemmatizados = []
            for noticia in noticias:
                tokens_originais.extend(noticia['tokens_originais'] )
                tokens_normalizados.extend(noticia['tokens_normalizados'] )
                tokens_stemmed.extend(noticia['tokens_stemmed'] )
                tokens_lemmatizados.extend(noticia['tokens_lemmatizados'] )
            
            salvar_lista(tokens_originais, cDirRun+"tokens.txt")
            salvar_lista(tokens_normalizados, cDirRun+"normalizados.txt")
            salvar_lista(tokens_stemmed, cDirRun+"stems.txt")
            salvar_lista(tokens_lemmatizados, cDirRun+"lemmas.txt")
            
            if noticias:
                print(f"\n=== SALVANDO DADOS PROCESSADOS ===")
                # Salva os dados em CSV
                arquivo_salvo = salvar_csv(noticias,cDirRun)
                
                if arquivo_salvo:
                    print(f"\n Processo concluído com sucesso!")
                    print(f"Arquivo gerado: {arquivo_salvo}")
                    print(f"\n  O arquivo contém dados de todas as etapas do processamento:")
                    print(f"   • Texto original e metadados")
                    print(f"   • Estatísticas de tokenização")
                    print(f"   • Resultados de stemming e lematização")
                    print(f"   • Métricas de qualidade linguística")
                else:
                    print("\n Erro ao salvar o arquivo CSV.")
            else:
                print("\n Nenhuma notícia foi extraída.")
        else:
            print("\n Nenhum link de notícia foi encontrado.")
        
        driver.quit()
    else:
        print(" Falha ao acessar o site.")

if __name__ == "__main__":
    main()