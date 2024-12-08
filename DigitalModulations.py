import numpy as np
import matplotlib.pyplot as plt

#####################################################################################################################################

# Função para gerar a codificação de Gray
def getGrayCode(M_order):
    """
    Gera o código Gray para M níveis.
    
    Parâmetros:
    - M_order (int): Número de níveis na constelação.
    
    Retorna:
    - gray_codes (dict): Mapeamento de números binários para Gray.
    """
    if not (M_order & (M_order - 1)) == 0:  # Verifica se M é potência de 2
        raise ValueError("M deve ser uma potência de 2.")
    
    gray_codes = {} # Dicionário para mapear binário para Gray
    for i in range(M_order): 
        # Para fazer a conversão de binário para Gray, basta fazer um XOR entre o número e o número deslocado uma posição para a direita (>> 1) 
        gray = i ^ (i >> 1)  # Cálculo do código Gray
        gray_codes[i] = gray 
    
    return gray_codes

#####################################################################################################################################

# Função para gerar a constelação M-PAM
def getPAM(bits:np.ndarray, m_order: int, c_distance: float):
    ''' 
    Realiza a modulação M-PAM com codificação Gray.

    Parâmetros:
    - bits (array): Sequência de bits a ser modulada.
    - m_order (int): Número de níveis na constelação (M-PAM).
    - c_distance (float): Distância mínima entre níveis.

    Retorna:
    - symbols (array): Símbolos modulados em M-PAM. 
    '''
    if not np.log2(m_order).is_integer():
        raise ValueError('O número de níveis da constelação deve ser uma potência de 2.')
    
    # Definindo o número de bits por símbolo
    K_BitsPerSymbol = int(np.log2(m_order))
    if len(bits) % K_BitsPerSymbol != 0:
        raise ValueError("O número de bits deve ser múltiplo de log2(M).")
    
    # Gerando o código Gray
    gray_codes = getGrayCode(m_order)
    gray_map = {i: gray_codes[i] for i in range(m_order)}  # Índice para código Gray

    # Gerando os níveis da constelação (usando índices e escalando com c_distance)
    pam_levels = np.array([2 * i - (m_order - 1) for i in range(m_order)]) * c_distance

    # Mapeamento do código Gray para níveis da constelação
    mapping = {tuple(map(int, format(gray_map[i], f'0{K_BitsPerSymbol}b'))): pam_levels[i] for i in range(m_order)}

    # Modulação dos bits
    symbols = []
    for i in range(0, len(bits), K_BitsPerSymbol):
        bit_chunk = tuple(bits[i:i + K_BitsPerSymbol])
        if bit_chunk not in mapping:
            raise KeyError(f"Bit chunk {bit_chunk} não encontrado no mapeamento.")
        symbols.append(mapping[bit_chunk])


    return np.array(symbols)

#####################################################################################################################################

# Função para gerar a constelação M-QAM
def getQAM(bits: np.ndarray, m_order: int, c_distance: float):
    ''' 
    Realiza a modulação M-QAM com codificação Gray.

    Parâmetros:
    - bits (array): Sequência de bits a ser modulada.
    - m_order (int): Número de níveis na constelação (Q-PAM).
    - c_distance (float): Distância mínima entre níveis.

    Retorna:
    - symbols (array): Símbolos modulados em Q-PAM.
    - gray_labels (dict): Mapeamento de cada símbolo para o código Gray correspondente.
    '''
    if not np.log2(m_order).is_integer():
        raise ValueError("O número de níveis da constelação deve ser uma potência de 2.")
    
    sqrt_m = int(np.sqrt(m_order)) # Número de níveis em cada dimensão da constelação, que será utilizado para fazer √M-PAM
    if sqrt_m**2 != m_order:
        raise ValueError("O número de níveis da constelação deve ser um quadrado perfeito.")
    
    # Definindo o número de bits por símbolo e dimensão
    K_BitsPerDimention = int(np.log2(sqrt_m))
    K_BitsPerSymbol = 2*K_BitsPerDimention
    if len(bits) % K_BitsPerSymbol != 0:
        raise ValueError("O número de bits deve ser múltiplo de 2*log2(M).")
    
    symbols = []
    gray_labels = {}
    for i in range(0, len(bits), K_BitsPerSymbol):
        in_phase_bits = bits[i:i+K_BitsPerDimention]
        quadrature_bits = bits[i+K_BitsPerDimention:i+K_BitsPerSymbol]

        # Fazendo a modulação √M-PAM
        in_phase = getPAM(in_phase_bits, sqrt_m, c_distance)[0] # Modulação PAM na dimensão I (in-phase) da constelação 
        quadrature = getPAM(quadrature_bits, sqrt_m, c_distance)[0] # Modulação PAM na dimensão Q (quadrature) da constelação

        # Criando agora a constelação a representação complexa de cada símbolo
        symbol = in_phase + 1j*quadrature
        symbols.append(symbol)

        # Convertendo os bits para string e mapeando para o código Gray, a partir da concatenação dos bits de cada dimensão (I e Q)
        gray_labels[symbol] = ''.join(map(str, in_phase_bits)) + ''.join(map(str, quadrature_bits))

    return np.array(symbols), gray_labels

#####################################################################################################################################





