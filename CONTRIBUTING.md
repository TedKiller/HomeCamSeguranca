# Guia de Contribuição — HomeCam Segurança

Ficamos muito felizes com o seu interesse em contribuir com o **HomeCam Segurança**! Este guia detalha o fluxo completo para clonar, configurar o ambiente, validar alterações e enviar um Pull Request (PR).

---

## 1. Pré-requisitos de Desenvolvimento

- **Python 3.11+** (`python --version`)
- **Git 2.40+**
- **FFmpeg** acessível no `PATH` ou via `imageio-ffmpeg` (instalado via `requirements.txt`)
- Windows 10/11 (opcional para DirectML GPU e testes GUI Tkinter) ou Linux c/ `python3-tk` (opcional apenas para rodar testes GUI)

---

## 2. Como Fazer Fork, Clone e Configurar o Ambiente

### Passo 1 — Fork no GitHub
Clique no botão **Fork** no canto superior direito da página do repositório oficial no GitHub para criar uma cópia na sua conta.

### Passo 2 — Clone do seu Fork
```bash
git clone https://github.com/SEU-USUARIO/HomeCamSeguranca.git
cd HomeCamSeguranca
git remote add upstream https://github.com/OWNER-ORIGINAL/HomeCamSeguranca.git
```

### Passo 3 — Criar Ambiente Virtual e Instalar Dependências
**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item config.example.json config.json
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
grep -vi "onnxruntime-directml" requirements.txt > /tmp/req-dev.txt
echo "onnxruntime>=1.18.0" >> /tmp/req-dev.txt
pip install -r /tmp/req-dev.txt
cp config.example.json config.json
```

---

## 3. Fluxo de Branch e Padrão de Commits

Sempre crie uma branch isolada a partir de `main` atualizada:

```bash
git checkout main
git pull upstream main
git checkout -b feature/nova-melhoria
# ou para correções de bugs:
# git checkout -b fix/corrige-reconexao-rtsp
```

Sugerimos prefixos claros no estilo **Conventional Commits**:
- `feat:` nova funcionalidade (ex.: `feat: adiciona suporte a câmera RTSP Reolink`)
- `fix:` correção de bug (ex.: `fix: evita trava ao alternar zoom na timeline`)
- `docs:` atualizações de documentação (`README.md`, `CONTRIBUTING.md`)
- `perf:` otimização de inferência, decodificação FFmpeg ou memória
- `test:` novos testes automatizados em `tests/`
- `refactor:` refatoração estrutural sem mudança de comportamento externo

---

## 4. Diretrizes de Código e Segurança

1. **Zero dados sensíveis no Git:** nunca adicione `config.json`, `.env`, `secrets/`, `gravacoes/`, `models/*.onnx` ou imagens com credenciais/chaves.
2. **Desacoplamento Headless / GUI:** qualquer lógica de captura, gravação, IA, timeline DVR ou acesso remoto deve permanecer livre de `tkinter` nos módulos de backend (`headless_nvr.py`, `web_portal.py`, `ai_analytics.py`, `dvr_playback.py`, `recorder.py`, `gdrive_uploader.py`, `camera_discovery.py`).
3. **Sem duplicação de lógica:** prefira extrair funções utilitárias compartilhadas entre GUI Desktop e serviço Headless 24/7.
4. **Cobertura de testes:** adicione ou atualize testes unitários/integração em `tests/test_app.py` (usando mocks para rede/ONNX/câmera quando apropriado para manter a suíte rápida e determinística).

---

## 5. Validação Obrigatória Antes do Pull Request

Execute todos os comandos abaixo dentro da raiz do projeto antes de enviar seu commit:

```powershell
# 1. Compilação sintática dos módulos
.\.venv\Scripts\python.exe -m py_compile main.py config_manager.py app_gui.py headless_nvr.py web_portal.py ai_analytics.py dvr_playback.py recorder.py gdrive_uploader.py ptz_control.py camera_discovery.py xiongmai_netip.py

# 2. Suíte completa de testes automatizados
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# 3. Smoke checks de inicialização (Desktop GUI e Servidor Headless 24/7)
.\.venv\Scripts\python.exe main.py --check
.\.venv\Scripts\python.exe main.py --headless --check
```

---

## 6. Abrindo o Pull Request (PR)

1. Envie sua branch para o seu fork:
   ```bash
   git add .
   git status  # confira se nenhum arquivo sensível ou pesado foi incluído
   git commit -m "feat: descreva brevemente sua melhoria"
   git push -u origin feature/nova-melhoria
   ```
2. Acesse seu repositório no GitHub e clique em **Compare & pull request**.
3. Preencha o resumo do PR informando:
   - Problema resolvido ou recurso adicionado;
   - Como testar (passo a passo);
   - Resultado da execução de `unittest discover -s tests -v` e dos smoke checks;
   - Prints de tela, caso haja alteração visual na GUI Desktop ou no Portal Web.
