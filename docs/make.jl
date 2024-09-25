using WholeGenomeModel
using Documenter

DocMeta.setdocmeta!(WholeGenomeModel, :DocTestSetup, :(using WholeGenomeModel); recursive=true)

makedocs(;
    modules=[WholeGenomeModel],
    authors="Olivier Labayle",
    sitename="WholeGenomeModel.jl",
    format=Documenter.HTML(;
        canonical="https://olivierlabayle.github.io/WholeGenomeModel.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/olivierlabayle/WholeGenomeModel.jl",
    devbranch="main",
)
