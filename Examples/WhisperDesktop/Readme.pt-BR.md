# WhisperDesktop — Português do Brasil

Esta versão traduz a interface do WhisperDesktop para português do Brasil, preservando a implementação de transcrição do fork sffxzzp/WhisperDesktop, inclusive o suporte a `ggml-large-v3-turbo-q5_0.bin`.

## Uso

1. Extraia `WhisperDesktop-Q5-PTBR.zip`.
2. Mantenha `WhisperDesktop.exe` e `Whisper.dll` na mesma pasta.
3. Abra `WhisperDesktop.exe` e selecione seu modelo GGML.
4. Para áudio em português, selecione `Portuguese`. Para legendas SRT, selecione `Legendas SubRip`.

O idioma da interface é independente do idioma de transcrição. `Traduzir` converte a fala para inglês. Os nomes de idiomas fornecidos pela API da biblioteca permanecem no idioma original. As preferências continuam compartilhadas com outras builds do WhisperDesktop.

## Arquivos traduzidos

Todos os caminhos abaixo são relativos a `Examples/WhisperDesktop`:

- `WhisperDesktop.rc`: títulos, rótulos, botões e links; idioma dos recursos e metadados `0x0416`; fonte Segoe UI 9 e ampliação de alguns rótulos/controles para acomodar os textos.
- `LoadModelDlg.cpp`: carregamento, validação e seleção do modelo.
- `TranscribeDlg.cpp`: formatos de saída, descrição do modelo, confirmações, erros e conclusão.
- `CaptureDlg.cpp`: gravação, dispositivos, seleção da saída e botões.
- `ModelAdvancedDlg.cpp`: opções avançadas de GPU.
- `Utils/DebugConsole.cpp`: título do console e mensagem de erro.
- `Utils/miscUtils.cpp`: mensagens gerais e aviso do recurso Traduzir.

As mudanças C++ se limitam a literais de texto. Os IDs dos recursos, estilos, posições dos campos de entrada e dimensões dos diálogos foram preservados. A descrição do modelo foi encurtada para caber no cabeçalho, mantendo nome, tamanho e implementação. O arquivo RC mantém sua codificação UTF-16 original.

## Compilação no Windows 10 x64

Instale o Visual Studio 2022 Build Tools com C++ x64/x86 (MSVC v143), ATL e Windows SDK, além do SDK .NET 10. A primeira execução de CompressShaders precisa restaurar o pacote `K4os.Compression.LZ4` 1.3.5.

No Developer PowerShell, a partir da raiz do repositório:

```powershell
$repoDir = (Get-Location).Path
msbuild ComputeShaders\ComputeShaders.vcxproj /m /t:Build /p:Configuration=Release /p:Platform=x64 /p:PlatformToolset=v143 "/p:SolutionDir=$repoDir\"
dotnet run --project Tools\CompressShaders\CompressShaders.csproj --configuration Release
msbuild Examples\WhisperDesktop\WhisperDesktop.vcxproj /m /t:Build /p:Configuration=Release /p:Platform=x64 /p:PlatformToolset=v143 "/p:SolutionDir=$repoDir\"
```

Verifique o código de saída de cada comando antes de prosseguir. A DLL e o executável ficam em `x64\Release`. O fork declara v145 nos projetos; a opção v143 permite usar as ferramentas do Visual Studio 2022 sem editar os arquivos de projeto.

Compile e compacte os shaders atuais: os shaders quantizados são necessários para preservar o suporte Q5. CompressShaders regenera `Whisper/D3D/shaderNames.cpp`, `Whisper/D3D/shaderNames.h`, `WhisperNet/API/eLanguage.cs` e os dados de shaders/tabelas. Esses arquivos gerados não fazem parte do patch de tradução.

## Validação da build PT-BR

- Base: `d12fddc9c2b2299182b4f33582055c0e94aef0a5`.
- Windows 10 x64, build 19045; MSVC 14.44.35207/v143; Windows SDK 10.0.26100.0; .NET SDK 10.0.204.
- Release x64 compilada; 57 shaders compactados, incluindo os shaders Q5.
- EXE com metadados de idioma `Português (Brasil)` e os quatro títulos de diálogos em PT-BR.
- IDs, estilos e campos de entrada comparados com a base; alterações C++ verificadas como exclusivamente literais.
- Modelo `ggml-large-v3-turbo-q5_0.bin` carregado na NVIDIA GeForce GTX 960M, Direct3D 11.0: vocabulário de 51.866 tokens e 587 tensores na GPU.
- Um trecho de 15 segundos gerou SRT com o exemplo `main` do próprio fork, usando uma cópia de hash idêntico à DLL distribuída. Processo concluído com código 0. Este teste valida carregamento e execução, não a qualidade da transcrição.
- O EXE iniciou e abriu a janela de transcrição. As medidas de texto foram verificadas com Segoe UI 9. A captura de janela falhou no ambiente de teste; a revisão visual completa das quatro telas e em diferentes escalas de exibição permanece pendente.

Licença: Mozilla Public License 2.0, conforme `LICENSE` na raiz. Tradução contribuída por `marcelolaranja13-lang`.
