#automatiza a tarefa de clonar o repositório nerd_tools
#o primeiro argumento traz um dos arquivos em ./templates para o projeto, enquanto o segundo determina o nome do arquivo no novo projeto.
#para adicionar múltiplos arquivos, basta rodar o script múltiplas vezes.

git clone https://github.com/almeida2808/nerd_tools.git

#cria um repositório na pasta do projeto com um .gitignore incluindo a pasta do ncdj_tools
echo "nerd_tools/" > .gitignore
git init

if [ "$#" -eq  "0" ]; then
    echo "Projeto vazio criado"
else
    if [ "$1" == "Python" ]; then
        cp ./nerd_tools/templates/Python.ipynb "$2"
    fi
    if [ "$3" == "start" ]; then
        jupyter notebook
    fi
fi