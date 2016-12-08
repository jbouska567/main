syntax enable
set tabstop=4
set shiftwidth=4
set expandtab
set softtabstop=4
set smarttab
set wildmenu
set ruler<
set hlsearch
set ignorecase
set smartcase
set number
set tabpagemax=30

set smartindent
set comments=s1:/*,mb:*,ex:*/

autocmd BufRead *.py set smartindent cinwords=if,elif,else,for,while,try,except,finally,def,class
set backspace=indent,eol,start
set list
set listchars=tab:>-
highlight SpecialKey guifg=yellow ctermfg=yellow

set wildmode=longest:full

colorscheme delek
set laststatus=2

highlight ExtraWhitespace ctermbg=red guibg=red
match ExtraWhitespace /\s\+$/
autocmd BufWinEnter * match ExtraWhitespace /\s\+$/
autocmd InsertEnter * match ExtraWhitespace /\s\+\%#\@<!$/
autocmd InsertLeave * match ExtraWhitespace /\s\+$/
autocmd BufWinLeave * call clearmatches()

set pastetoggle=<F10>

" Opraveni kurzorovych klavesech pro tmux
if &term =~ '^screen'
    " tmux will send xterm-style keys when its xterm-keys option is on
    execute "set <xUp>=\e[1;*A"
    execute "set <xDown>=\e[1;*B"
    execute "set <xRight>=\e[1;*C"
    execute "set <xLeft>=\e[1;*D"
    execute "set <PageUp>=\e[5;*~"
    execute "set <PageDown>=\e[6;*~"
    execute "set <xHome>=\e[1;*H"
    execute "set <xEnd>=\e[1;*F"
endif

" prechod mezi taby ctrl-pgup/down
map <C-pageup> :tabp<CR>
map <C-pagedown> :tabn<CR>

map <C-\> :tab split<CR>:exec("tag ".expand("<cword>"))<CR>
map <A-\> :vsp <CR>:exec("tag ".expand("<cword>"))<CR>
set tags=~/tags;~/git/tags
