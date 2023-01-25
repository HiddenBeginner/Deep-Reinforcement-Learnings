Search.setIndex({docnames:["book/Chapter1/1-sequential-decision-making-problems","book/Chapter1/2-markov-decision-processes","book/Chapter1/3-policy-return-value","book/Chapter1/4-bellman-equation","book/Chapter1/5-stochastic-approximation","book/Chapter2/1-policy-gradient-theorem","book/Chapter2/2-reinforce","book/Reference","book/intro"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":5,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,"sphinxcontrib.bibtex":9,sphinx:56},filenames:["book/Chapter1/1-sequential-decision-making-problems.md","book/Chapter1/2-markov-decision-processes.md","book/Chapter1/3-policy-return-value.md","book/Chapter1/4-bellman-equation.md","book/Chapter1/5-stochastic-approximation.md","book/Chapter2/1-policy-gradient-theorem.md","book/Chapter2/2-reinforce.md","book/Reference.md","book/intro.md"],objects:{},objnames:{},objtypes:{},terms:{"0":[1,2,4],"0\uacfc":[1,2],"0\ubcf4\ub2e4":2,"0\uc744":2,"0\uc774":3,"0\uc778":1,"1":[1,2,3,4,7,8],"100m":1,"180":1,"1861":7,"1870":7,"1\ubc18\uc758":3,"1\uc0ac\uc774\uc758":[1,2],"1\uc758":2,"1\ud559\ub144":3,"1step":3,"1step_return":[],"1step_state_value_funct":[],"2":[1,2,3,4],"2018":7,"2\uac00\uc9c0":3,"2\uac1c\ub97c":3,"2step":3,"3":[1,2,3],"35th":7,"360":1,"3\uc885\ub958\uac00":2,"4":[3,4],"5":[3,4],"6":[1,3],"6\ucd95":1,"7":3,"80":7,"\uac00":[1,2,3],"\uac00\ub2a5\ud558\ub2e4":3,"\uac00\ub2a5\ud560\uaec4":3,"\uac00\uc18d\ub3c4\uac00":[],"\uac00\uc18d\ub3c4\ub294":1,"\uac00\uc7a5":[1,2,3],"\uac00\uc815\uc744":1,"\uac00\uc815\uc774":1,"\uac00\uc815\uc778\uc9c0":1,"\uac00\uc815\ud558\uae30":1,"\uac00\uc815\ud558\uc5ec":1,"\uac00\uc815\ud558\uc790":[1,3],"\uac00\uc815\ud55c\ub2e4":1,"\uac00\uc911\uce58\ub97c":[1,2],"\uac00\uc911\ud569\uc744":1,"\uac00\uc9c0":[2,4],"\uac00\uc9c0\ub294":1,"\uac00\uc9c0\ub85c":[],"\uac00\uc9c0\ub97c":1,"\uac00\uc9c8":1,"\uac01":[1,2,3],"\uac01\uac01":1,"\uac01\ub3c4\uac00":1,"\uac01\ub3c4\ub85c":1,"\uac01\uc790":2,"\uac04\ub2e8\ud558\uace0":4,"\uac04\ub2e8\ud558\uc9c0\ub9cc":3,"\uac04\uc8fc\ud558\uc5ec":2,"\uac10\uc0ac\ub4dc\ub9ac\uaca0\ub2e4":8,"\uac12":4,"\uac12\ub9cc":1,"\uac12\uc73c\ub85c":[1,4],"\uac12\uc744":1,"\uac12\uc774\ub77c\uace0":2,"\uac12\uc774\uba70":2,"\uac12\uc778":4,"\uac15\ud654":[2,4],"\uac15\ud654\ud559\uc2b5":[1,3,4,8],"\uac15\ud654\ud559\uc2b5\ub3c4":8,"\uac15\ud654\ud559\uc2b5\uc5d0\uc11c":2,"\uac15\ud654\ud559\uc2b5\uc5d0\uc11c\ub294":4,"\uac15\ud654\ud559\uc2b5\uc740":[0,8],"\uac15\ud654\ud559\uc2b5\uc744":4,"\uac15\ud654\ud559\uc2b5\uc774":8,"\uac16\uace0":[2,3],"\uac16\ub294":[1,2],"\uac16\ub2e4":8,"\uac19\ub2e4":[2,3,4],"\uac19\uc544\uc9c4\ub2e4":4,"\uac19\uc740":[1,2,3],"\uac19\uc774":[1,4],"\uac1c\ub150":[2,3],"\uac1c\ub150\uc774\ub2c8":2,"\uac1c\ub150\uc778":[],"\uac1c\uc120\ub41c":2,"\uac1c\uc120\ud558\ub294":2,"\uac1c\uc120\ud560":2,"\uac1c\uc218\uac00":4,"\uac1c\uc218\uc774\ub2e4":4,"\uac78\uc74c\uc740":3,"\uac83":8,"\uac83\uacfc":2,"\uac83\ub3c4":3,"\uac83\ub9cc\uc73c\ub85c\ub3c4":2,"\uac83\ubcf4\ub2e4":2,"\uac83\ubd80\ud130":3,"\uac83\uc5d0":1,"\uac83\uc73c\ub85c":2,"\uac83\uc740":[2,3,4],"\uac83\uc744":[0,1,2,3,4],"\uac83\uc774":[1,2,3,8],"\uac83\uc774\uace0":[1,2],"\uac83\uc774\uae30":[1,2],"\uac83\uc774\ub2e4":[0,1,2,3,4,8],"\uac83\uc778\uac00":2,"\uac83\uc778\ub370":3,"\uac83\uc77c\uae4c":1,"\uac83\ucc98\ub7fc":[0,3],"\uac89\uc73c\ub85c":2,"\uac8c\uc784\uc758":1,"\uac8c\uc784\uc774":2,"\uacb0\uad6d":4,"\uacb0\uc815":[1,2],"\uacb0\uc815\ub418\uae30":2,"\uacb0\uc815\ub418\uc5c8\uae30":2,"\uacb0\uc815\ub418\uc5c8\ub2e4":2,"\uacb0\uc815\ub41c\ub2e4":[1,3],"\uacb0\uc815\ub41c\ub2e4\uace0":1,"\uacb0\uc815\uc744":0,"\uacb0\uc815\uc801":2,"\uacbd\uc6b0":[1,2,3],"\uacbd\uc6b0\uac00":4,"\uacbd\uc6b0\uc774\ub2e4":2,"\uacbd\ud5d8\ud574\ubcf4\uc558\uc73c\ub2c8":8,"\uacc4\uc0b0\ub418\uae30":3,"\uacc4\uc0b0\ub41c\ub2e4":3,"\uacc4\uc0b0\uc5d0\uc11c":3,"\uacc4\uc0b0\uc774":[3,4],"\uacc4\uc0b0\uc774\uace0":[],"\uacc4\uc0b0\ud558\uae30":1,"\uacc4\uc0b0\ud558\ub294":[2,3,4],"\uacc4\uc0b0\ud560":[1,2,4],"\uacc4\uc18d":4,"\uace0\ub824\ud558\ub294":2,"\uace0\ub824\ud558\uba70":2,"\uace0\ub824\ud558\uc9c0":2,"\uace0\ub824\ud55c\ub2e4":2,"\uace0\ub824\ud560":2,"\uace0\ub824\ud574\uc57c":1,"\uace0\uc815\ub418\uc5c8\ub2e4\ub294":3,"\uace0\uc815\ub418\uc5c8\uc73c\ub2c8":3,"\uace0\uc815\ud558\uc5ec":4,"\uacf1\ud574":3,"\uacf1\ud574\uc900":3,"\uacf1\ud574\uc900\ub2e4":3,"\uacf1\ud574\uc9c4":[],"\uacf5\uac04":3,"\uacf5\uac04\uacfc":3,"\uacf5\uac04\uc744":1,"\uacf5\uac04\uc758":3,"\uacf5\uac04\uc774\ub77c":3,"\uacf5\ubd80\ud558\ub294":8,"\uacf5\ubd80\ud558\uba74":8,"\uacf5\ubd80\ud55c":2,"\uacf5\ubd80\ud560":[4,8],"\uacf5\ubd80\ud574\ubcf4\uc790":2,"\uacf5\ubd80\ud574\uc628":8,"\uacf5\ud1b5\uc801\uc73c\ub85c":3,"\uacfc":[2,4],"\uacfc\uac70":1,"\uacfc\uac70\uc758":1,"\uacfc\uc815":1,"\uacfc\uc815\uc744":0,"\uacfc\uc815\uc758":1,"\uad00\uacc4\ub97c":[2,3,4],"\uad00\uacc4\uc2dd\ub3c4":3,"\uad00\uacc4\uc2dd\uc774":2,"\uad00\uc808":1,"\uad00\uc808\ub9c8\ub2e4":1,"\uad00\uc808\uc740":1,"\uad00\uc808\uc744":1,"\uad00\uc808\uc758":1,"\uad00\uc808\uc774":1,"\uad00\uc810\uc5d0\uc11c":4,"\uad00\uce21":4,"\uad00\uce21\uac12":4,"\uad00\uce21\uac12\uc740":4,"\uad00\uce21\uac12\uc778":2,"\uad00\uce21\uc744":4,"\uad00\uce21\ud558\uace0":0,"\uad49\uc7a5\uc774":2,"\uad49\uc7a5\ud788":[1,2,3,4],"\uad6c\ubcc4\ud560":4,"\uad6c\ubd84\ub420":2,"\uad6c\uc870\uac00":8,"\uad6c\ud558\uace0":3,"\uad6c\ud558\ub294":4,"\uad6c\ud560":3,"\uad6c\ud604\uc5d0\uc11c\ub294":4,"\uada4\ub3c4":1,"\uada4\uc801\uc778\ub370":1,"\uaddc\uce59\uc5d0":1,"\uadf8":[0,8],"\uadf8\ub0e5":2,"\uadf8\ub300\ub85c":[1,4,8],"\uadf8\ub798\uc11c":8,"\uadf8\ub7ec\uae30":[],"\uadf8\ub7fc":[0,2],"\uadf8\ub807\uc9c0":8,"\uadf8\ub808\ub514\uc5b8\ud2b8\ub97c":2,"\uadf8\ub9ac\uace0":[1,2,4],"\uadf8\uc800":8,"\uadf9\uc801\uc73c\ub85c":2,"\uadfc\uac70\uac00":3,"\uadfc\uc0ac\uc2dc\ud0a4\ub294":4,"\uae30\ub300\ud560":2,"\uae30\ub313\uac12":[2,3,4],"\uae30\ub313\uac12\uc5d0":[3,4],"\uae30\ub313\uac12\uc73c\ub85c":[2,4],"\uae30\ub313\uac12\uc73c\ub85c\uc11c":3,"\uae30\ub313\uac12\uc740":3,"\uae30\ub313\uac12\uc744":2,"\uae30\ub313\uac12\uc758":[2,3,4],"\uae30\ub313\uac12\uc774":3,"\uae30\ub313\uac12\uc774\uace0":[2,3],"\uae30\ub313\uac12\uc774\ub2e4":[2,3],"\uae30\ub313\uac12\uc774\ub77c\ub294":3,"\uae30\uc220\ud558\uae30":1,"\uae30\uc220\ud55c\ub2e4":1,"\uae30\uc220\ud560":1,"\uae30\uc5b5\ud558\uba74":4,"\uae30\uc874":4,"\uae30\uc900\uc744":2,"\uae43\ud5d9":8,"\uae4c\uc9c0":3,"\uae4c\uc9c0\uac00":[],"\uae4c\uc9c0\ub9cc":2,"\uaf2d":[2,8],"\uaf3c\uaf3c\ud558\uac8c":8,"\ub05d\ub09c\ub2e4":2,"\ub05d\uc774":2,"\ub098":3,"\ub098\uac08":8,"\ub098\uba38\uc9c0":2,"\ub098\uc05c":2,"\ub098\uc068\uc744":[1,2,3],"\ub098\uc628":4,"\ub098\uc628\ub2e4":4,"\ub098\uc62c":[2,3],"\ub098\uc704":8,"\ub098\uc758":8,"\ub098\uc911\uc5d0":[1,3],"\ub098\ud0c0\ub098\ub294":4,"\ub098\ud0c0\ub09c":4,"\ub098\ud0c0\ub0b4\uace0":1,"\ub098\ud0c0\ub0b4\uae30":2,"\ub098\ud0c0\ub0b4\ub294":1,"\ub098\ud0c0\ub0b8":[],"\ub098\ud0c0\ub0b8\ub2e4":[1,2,3],"\ub098\ud0c0\ub0bc":3,"\ub0a8\uaca8\uc8fc\uba74":8,"\ub0a8\uaca8\uc8fc\uc2dc\uba74":8,"\ub0a8\uae38":8,"\ub0a8\uc544":2,"\ub0ae\uc740":[1,2],"\ub0b4\uac00":1,"\ub0b4\ub294":3,"\ub0b4\uc11c":3,"\ub0b4\uc6a9":8,"\ub0b4\uc6a9\ub4e4\ub3c4":8,"\ub0b4\uc6a9\uc744":8,"\ub0b4\ud3ec\ub418\uc5b4":2,"\ub0b8":2,"\ub0c8\uc744\uc9c0\ub3c4":2,"\ub108\ubb34":[1,2],"\ub123\uc5b4\uc8fc\ub294":3,"\ub123\uc5b4\uc8fc\uba74":3,"\ub123\uc73c\uba74":3,"\ub124\ud2b8\uc6cc\ud06c":[3,8],"\ub124\ud2b8\uc6cc\ud06c\ub85c":2,"\ub124\ud2b8\uc6cc\ud06c\ub97c":8,"\ub178\ub825\ud558\uc600\ub2e4":2,"\ub17c\ubb38":8,"\ub17c\ubb38\ub9c8\ub2e4":1,"\ub17c\ubb38\uc744":8,"\ub204\uc801":[1,2,3],"\ub208\uc5d0":3,"\ub274\ub7f4":[2,8],"\ub290\uaef4\uc9c4":8,"\ub294":[1,2,3,4],"\ub2e4":1,"\ub2e4\ub8e8\uace0":1,"\ub2e4\ub974\ub2e4":2,"\ub2e4\ub978":3,"\ub2e4\ub978\ub370":1,"\ub2e4\uc2dc":[0,2,3,4],"\ub2e4\uc74c":[0,1,2,3,4],"\ub2e4\uc74c\uacfc":[1,2,3,4],"\ub2e4\uc74c\uc73c\ub85c":[1,3],"\ub2e4\uc74c\uc744":2,"\ub2e4\uc74c\uc774":2,"\ub2e8":[1,8],"\ub2e8\uc21c\ud558\uac8c":1,"\ub2e8\uc21c\ud654\ub97c":1,"\ub2e8\uc21c\ud654\uc2dc\ud0a8\ub2e4":1,"\ub2e8\uc21c\ud654\ud558\uc5ec":1,"\ub2ec\ub9ac\uae30\ub97c":1,"\ub2f5\ubcc0\uc744":2,"\ub2f5\uc740":2,"\ub300\uad04\ud638":3,"\ub300\ubb38\uc790\ub85c":2,"\ub300\ubd80\ubd84\uc758":2,"\ub300\uc0c1\uc744":0,"\ub300\uc0c1\uc774\ub2e4":1,"\ub300\uc2e0":[1,2,4],"\ub300\uc751\ud558\ub294":1,"\ub300\uc751\ud55c\ub2e4":4,"\ub300\uc785\ud574\ubcf4\uc790":3,"\ub300\uccb4\ud574\uc11c":4,"\ub300\ud559\uc6d0":4,"\ub300\ud55c":[2,3,4],"\ub300\ud574":[1,2,3],"\ub300\ud574\uc11c":[1,2,3],"\ub300\ud574\uc11c\ub294":[1,3],"\ub300\ud574\uc11c\ub3c4":2,"\ub313\uae00\uc740":8,"\ub313\uae00\uc744":8,"\ub354":[1,2,3],"\ub354\ub7ec":8,"\ub354\ud558\uae30":3,"\ub354\ud560":8,"\ub354\ud574\uc8fc\ub294":1,"\ub354\ud574\uc900":3,"\ub355\ubd84\uc5d0":2,"\ub367\uc148":4,"\ub36e\uc5c8\ub2e4":8,"\ub370":8,"\ub370\uc774\ud130":4,"\ub370\uc774\ud130\uac00":4,"\ub370\uc774\ud130\uc758":[4,8],"\ub3c4\ub9cc\ud07c":1,"\ub3c4\uc640\uc8fc\ub294":8,"\ub3c4\uc6c0\uc774":8,"\ub3c5\uc790\uc758":8,"\ub3d9\uc548":8,"\ub410\uc5c8\ub294\ub370":8,"\ub418\uace0":0,"\ub418\ub294":[0,2,3],"\ub418\ub3c4\ub85d":3,"\ub418\uc5b4":[1,3],"\ub418\uc5c8\ub2e4":1,"\ub418\uc5c8\uc73c\uba74":8,"\ub41c\ub2e4":[1,2,3,4],"\ub420":[2,3],"\ub450":[2,3],"\ub4e4\uc5b4":[1,3],"\ub4e4\uc5b4\ub193\uc558\ub2e4":3,"\ub4f1":8,"\ub4f1\uc18d\uc6b4\ub3d9\uc774\ub098":[],"\ub4f1\uc2dd\uc774\uba70":3,"\ub4f1\uc73c\ub85c":3,"\ub525\ub7ec\ub2dd":8,"\ub525\ub7ec\ub2dd\uc5d0":2,"\ub525\ub7ec\ub2dd\uc5d0\uc11c":2,"\ub525\ub7ec\ub2dd\uc744":[2,8],"\ub525\ub7ec\ub2dd\uc758":8,"\ub530\ub77c":[2,3],"\ub530\ub77c\uc11c":[1,2,3],"\ub530\ub790\ub2e4\ub294":2,"\ub530\ub790\uc744":[2,3],"\ub530\ub85c":1,"\ub530\ub974\ub294":[2,3],"\ub530\ub974\ub294\uc9c0":2,"\ub530\ub974\uba70":2,"\ub530\ub974\uc9c0":2,"\ub530\ub978\ub2e4":1,"\ub531":[0,1,2,3],"\ub54c":[1,2,3,4,8],"\ub54c\ub294":[2,8],"\ub54c\ub9c8\ub2e4":4,"\ub54c\ubb38\uc5d0":[1,2,3,4,8],"\ub54c\ubb38\uc774\ub2e4":[1,2],"\ub54c\uc640":4,"\ub550":4,"\ub610":0,"\ub610\ub294":[0,4],"\ub610\ud55c":4,"\ub73b\uc774":[],"\ub77c":2,"\ub77c\uace0":[0,1,2,3],"\ub85c":[1,2,3,4],"\ub85c\ubd07":1,"\ub85c\ubd07\uc758":1,"\ub97c":[0,1,2,3,4],"\ub9c8\ub9ac\uc624":1,"\ub9c8\ucc2c\uac00\uc9c0\ub85c":4,"\ub9cc":[2,3,4],"\ub9cc\ub4e4\uc5b4":2,"\ub9cc\ub4e4\uc5b4\ub0b8":1,"\ub9cc\ub4e4\uc5b4\uc8fc\ub294":[1,2],"\ub9cc\ub4e4\uc5b4\uc9c4":2,"\ub9cc\uc73c\ub85c":2,"\ub9cc\uc871\ud558\ub294":[2,3],"\ub9cc\uc871\ud55c\ub2e4\uace0":1,"\ub9cc\uc871\ud55c\ub2e4\ub294":[],"\ub9cc\uc871\ud574\uc57c":3,"\ub9ce\ub2e4":[2,4],"\ub9ce\uc740":[2,3],"\ub9ce\uc744":4,"\ub9ce\uc774":[0,1],"\ub9d0":1,"\ub9d0\uace0":8,"\ub9d0\ud55c\ub2e4":[1,2],"\ub9d0\ud574\uc900\ub2e4":3,"\ub9e4":[0,2],"\uba3c":2,"\uba3c\uc800":[1,2,8],"\uba87":1,"\uba87\uc774\uc5c8\ub294\uc9c0":1,"\ubaa8\ub378\ub9c1\ud55c\ub2e4\uace0":1,"\ubaa8\ub378\ub9c1\ud560":1,"\ubaa8\ub450":[2,3,4],"\ubaa8\ub4e0":[1,2,3,4],"\ubaa8\ub974\uaca0\uc9c0\ub9cc":1,"\ubaa8\ub978\ub2e4":2,"\ubaa8\uc2b5\uc774\ub2e4":3,"\ubaa9\uc801":2,"\ubaa9\uc801\uc73c\ub85c":8,"\ubaa9\uc801\ud568\uc218\uac00":2,"\ubaa9\uc801\ud568\uc218\ub97c":2,"\ubaa9\ud45c\ub294":[0,1,2,8],"\ubab0\ub790\uc744":8,"\ubabb\ud558\uace0":8,"\ubabb\ud55c":0,"\ubabb\ud588\uae30":4,"\ubabb\ud588\ub2e4":4,"\ubb34\uc5c7\uc744":2,"\ubb34\uc5c7\uc774\uace0":1,"\ubb34\uc5c7\uc774\uba70":1,"\ubb34\uc5c7\uc778\uc9c0":[2,8],"\ubb34\uc5c7\uc77c\uae4c":[],"\ubb34\uc9c4\uc7a5":4,"\ubb34\ud2bc":4,"\ubb34\ud55c\ud788":[0,2],"\ubb36\uc5b4\uc8fc\uc790":3,"\ubb38\uc5b4\uccb4\uc640":8,"\ubb38\uc7a5\ub3c4":8,"\ubb38\uc81c":1,"\ubb38\uc81c\uac00":[1,2],"\ubb38\uc81c\ub294":[0,1],"\ubb38\uc81c\ub77c\uace0":0,"\ubb38\uc81c\ub97c":[0,1,2],"\ubb38\uc81c\uc5d0\uc11c":0,"\ubb38\uc81c\uc774\ub2e4":0,"\ubb3c\ub860":[3,4],"\ubbf8\ub798\uc5d0":2,"\ubbf8\ub9ac":3,"\ubbf8\ubd84":4,"\ubc08\uc744":8,"\ubc14\uae65":3,"\ubc14\ub00c\uac8c":0,"\ubc14\ub00c\uace0":0,"\ubc14\ub00c\ub294\uc9c0":1,"\ubc14\ub00c\uc5c8\uace0":2,"\ubc14\ub00c\uc9c0":2,"\ubc14\ub010":[0,1],"\ubc14\ub85c":[2,3,8],"\ubc14\ubcf4":2,"\ubc14\ud0d5\uc73c\ub85c":3,"\ubc15\uc2a4\uc5d0":3,"\ubc16\uc73c\ub85c":[3,4],"\ubc18":3,"\ubc18\ubcf5\uc801\uc73c\ub85c":4,"\ubc18\ubcf5\ud558\uac70\ub098":0,"\ubc18\ubcf5\ud558\ub294":0,"\ubc18\ubcf5\ud558\uc5ec":0,"\ubc18\uc5d0":3,"\ubc1b\uac8c":[2,3],"\ubc1b\ub294":2,"\ubc1b\uc544":2,"\ubc1b\uc558\uc744":0,"\ubc1b\uc558\uc744\ud14c\uace0":0,"\ubc1b\uc740":[0,1,2],"\ubc1b\uc744":2,"\ubc1c\uacac\ud558\uac70\ub098":8,"\ubc1c\uc0dd\ud560":3,"\ubc1c\uc0dd\ud560\uc9c0":1,"\ubc29\ubb38\ud560":3,"\ubc29\ubc95\ub860\uc774\ub2e4":[0,4],"\ubc29\ubc95\ub860\uc778":3,"\ubc29\ubc95\uc5d0":2,"\ubc29\ubc95\uc744":4,"\ubc29\uc2dd\uc73c\ub85c":4,"\ubc29\uc815\uc2dd\uacfc":4,"\ubc29\uc815\uc2dd\uc5d0":3,"\ubc29\uc815\uc2dd\uc744":3,"\ubc29\uc815\uc2dd\uc758":3,"\ubc29\uc815\uc2dd\uc774":3,"\ubc29\uc815\uc2dd\uc774\ub77c\uace0":3,"\ubc29\ud5a5\uc73c\ub85c":2,"\ubc84\uc804\ubd80\ud130":3,"\ubc88":1,"\ubc88\uc529":8,"\ubc88\uc5ed\ud558\uc9c0":8,"\ubc88\uc9f8":[1,3,4],"\ubc95":8,"\ubc95\uce59\uc5d0":3,"\ubc95\uce59\uc5d0\uc11c":3,"\ubc95\uce59\uc744":3,"\ubc95\uce59\uc774":3,"\ubcc0\uacbd\ud558\uba70":1,"\ubcc0\uacbd\ud568\uc73c\ub85c\uc11c":2,"\ubcc0\uc218":[2,3,4],"\ubcc0\uc218\uac00":[2,3],"\ubcc0\uc218\ub294":3,"\ubcc0\uc218\ub4e4\uc774":2,"\ubcc0\uc218\ub97c":[2,3],"\ubcc0\uc218\uc5d0":[2,3],"\ubcc0\uc218\uc640":4,"\ubcc0\uc218\uc758":2,"\ubcf4\uace0\ub418\uba74\uc11c":8,"\ubcf4\ub0b4\ub294":1,"\ubcf4\ub2e4":2,"\ubcf4\uba74":[2,4],"\ubcf4\uc0c1":[0,2,3],"\ubcf4\uc0c1\uae4c\uc9c0":2,"\ubcf4\uc0c1\ub4e4\uc758":[0,2,3],"\ubcf4\uc0c1\ub4e4\uc774":2,"\ubcf4\uc0c1\uc740":2,"\ubcf4\uc0c1\uc744":[0,1],"\ubcf4\uc0c1\uc758":[1,2],"\ubcf4\uc0c1\uc774":2,"\ubcf4\uc0c1\uc77c\uc218\ub85d":[1,2],"\ubcf4\uc5ec\uc8fc\uba70":3,"\ubcf4\uc5ec\uc900\ub2e4":[3,4],"\ubcf4\uc774\ub294":[2,3],"\ubcf4\uc790":3,"\ubcf4\uc7a5\ud558\uae30":1,"\ubcf4\ud1b5":[2,4],"\ubcf4\ud3b8\uc801\uc778":1,"\ubcf8":[2,4],"\ubd10\ub450\uba74":4,"\ubd80\ub514":8,"\ubd80\ub958\uc5d0":2,"\ubd80\ub974\uae30\ub3c4":4,"\ubd80\ub974\ub294":1,"\ubd80\ub974\uba70":2,"\ubd80\ub978\ub2e4":[0,1,2,3,4],"\ubd80\ub97c":1,"\ubd80\ub984":4,"\ubd80\ubd84":1,"\ubd80\ubd84\uc740":3,"\ubd80\uc5ec\ub418\uc5b4":1,"\ubd80\uc5ec\ud558\ub294":1,"\ubd80\uc5ec\ud55c\ub2e4":[0,1],"\ubd80\uc5ec\ud560":1,"\ubd80\ud130":1,"\ubd84\ub4e4\uc774":8,"\ubd84\ub4e4\uc774\ub77c\uba74":2,"\ubd84\uba85":8,"\ubd84\uba85\ud558\ub2e4":1,"\ubd84\uc57c\uac00":8,"\ubd84\uc57c\ub294":8,"\ubd84\uc57c\ub97c":8,"\ubd84\uc57c\uc5d0":8,"\ubd84\uc57c\uc5d0\uc11c":[3,8],"\ubd84\uc57c\uc758":8,"\ubd84\uc57c\uc778\uc904":8,"\ubd84\uc57c\ucc98\ub7fc":8,"\ubd84\ud3ec":3,"\ubd84\ud3ec\ub85c":2,"\ubd84\ud3ec\ub85c\ubd80\ud130":2,"\ubd84\ud3ec\ub97c":[1,2,3],"\ubd84\ud3ec\uc5d0":2,"\ubd84\ud3ec\uc5d0\uc11c":1,"\ubd84\ud3ec\uc774\ub2e4":1,"\ubd84\ud3ec\uc784\uc744":2,"\ubd88\uac00\ub2a5":3,"\ubd88\uac00\ub2a5\ud558\ub2e4":2,"\ubd88\ud655\uc2e4\uc131\uc774":2,"\ube44\uad50\ud558\uc5ec":2,"\ube44\uad50\ud560":2,"\ube44\uc720\ud558\uac74\ub370":2,"\ube44\uc804":8,"\ube44\ud6a8\uc728\uc801\uc774\ub2e4":4,"\ube44\ud6a8\uc728\uc801\uc77c":4,"\ube60\ub974\uac8c":8,"\ube7c\ub0b4\uc11c":4,"\ube80":2,"\ubfd0\uc774\ub2e4":8,"\uc0ac\ub840\uac00":8,"\uc0ac\uc2e4":2,"\uc0ac\uc2e4\uc0c1":3,"\uc0ac\uc6a9\ub418\uae30":4,"\uc0ac\uc6a9\ub418\ub294":1,"\uc0ac\uc6a9\ub41c\ub2e4":3,"\uc0ac\uc6a9\ud558\ub294":[2,8],"\uc0ac\uc6a9\ud558\uc5ec":[2,3],"\uc0ac\uc6a9\ud55c":[2,3],"\uc0ac\uc6a9\ud55c\ub2e4":4,"\uc0ac\uc6a9\ud55c\ub2e4\uac70\ub098":8,"\uc0ac\uc6a9\ud560":[1,2,3,4],"\uc0ac\uc6a9\ud574\ub3c4":4,"\uc0ac\uc6a9\ud574\uc11c":3,"\uc0ac\uc774\uc758":[1,2,3],"\uc0b0\ucd9c\ubb3c\uc774\ub2e4":1,"\uc0c1\uc218":4,"\uc0c1\uc218\uac00":3,"\uc0c1\uc218\uc774\uc5b4\uc57c":4,"\uc0c1\ud0dc":[0,4],"\uc0c1\ud0dc\uac00":[0,1,2],"\uc0c1\ud0dc\ub294":[1,2],"\uc0c1\ud0dc\ub4e4\uc758":[1,3],"\uc0c1\ud0dc\ub85c":1,"\uc0c1\ud0dc\ub97c":[0,1,2],"\uc0c1\ud0dc\ub9c8\ub2e4":[1,2],"\uc0c1\ud0dc\ubd80\ud130":1,"\uc0c1\ud0dc\uc5d0":[0,2],"\uc0c1\ud0dc\uc5d0\uc11c":[1,2],"\uc0c1\ud0dc\uc640":1,"\uc0c1\ud0dc\uc758":[1,3],"\uc0c1\ud638\uc791\uc6a9\uc744":2,"\uc0c1\ud669\uc5d0\uc11c\ub294":4,"\uc0c1\ud669\uc778":2,"\uc0c8\ub85c\uc6b4":8,"\uc0d8\ud50c\ub9c1":4,"\uc0d8\ud50c\ub9c1\ub418\uc5c8\uace0":2,"\uc0d8\ud50c\ub9c1\ub418\uc5c8\ub2e4\ub294":1,"\uc0d8\ud50c\ub9c1\ub41c":2,"\uc0d8\ud50c\ub9c1\ub41c\ub2e4":3,"\uc0d8\ud50c\ub9c1\ud558\uba74\uc11c":4,"\uc0d8\ud50c\ub9c1\ud558\uc5ec":1,"\uc0dd\uac01\ud558\uba74":3,"\uc0dd\uac01\ud560":2,"\uc0dd\uac01\ud574\ubcf4\uba74":2,"\uc0dd\uac01\ud574\ubcf4\uc790":[1,2],"\uc0dd\uae30\uba74":8,"\uc0dd\ub7b5\ud574\uc8fc\uace0":4,"\uc120\ud0dd\uad8c\uc774":3,"\uc120\ud0dd\ub418\uc5c8\uc73c\uba70":2,"\uc120\ud0dd\ud558\uac8c":2,"\uc120\ud0dd\ud558\uae30":2,"\uc120\ud0dd\ud55c":3,"\uc120\ud0dd\ud560":2,"\uc124\uba85\uc774":8,"\uc131\uacf5\uc801\uc778":8,"\uc131\ub2a5\uc740":2,"\uc131\ub2a5\uc744":[2,3],"\uc131\ub2a5\uc774":8,"\uc131\ub9bd\ud55c\ub2e4\uace0":1,"\uc131\ub9bd\ud560":1,"\uc131\uc9c8\uc5d0":3,"\uc131\uc9c8\uc744":3,"\uc131\uc9c8\uc778":3,"\uc138":[2,4],"\uc138\uc6cc\uc57c":2,"\uc148\uc774\ub2e4":2,"\uc18c\ubb38\uc790\ub85c":2,"\uc18d\ub3c4\uac00":1,"\uc18d\ub3c4\ub294":1,"\uc18d\ub3c4\ub85c":1,"\uc18d\ub3c4\ub9cc":1,"\uc18d\ud558\uc9c0\ub294":2,"\uc218":[0,1,2,3,4,8],"\uc218\uac15\ud558\uc9c0":4,"\uc218\ub3c4":[1,2],"\uc218\ub834\uc131\uc5d0":4,"\uc218\ub834\uc131\uc744":1,"\uc218\ub834\uc131\uc774":4,"\uc218\ub834\ud55c\ub2e4\ub294":4,"\uc218\uc2dd\uc5d0\uc11c":4,"\uc218\uc2dd\uc801\uc73c\ub85c\ub294":2,"\uc218\uc815\ud558\ub294":2,"\uc218\uc900\uc758":4,"\uc218\uce58\ub85c":3,"\uc218\uce58\uc801\uc73c\ub85c":2,"\uc218\ud559\uc801\uc73c\ub85c":1,"\uc218\ud589\ud558\ub294":0,"\uc21c\uc11c":2,"\uc21c\uc11c\uc30d":1,"\uc21c\ucc28\uc801":[1,2],"\uc21c\ucc28\uc801\uc73c\ub85c":0,"\uc228\uc740":8,"\uc26c\uc6b4":3,"\uc27d\uac8c":[1,3,8],"\uc27d\ub2e4":4,"\uc288\ud37c":1,"\uc2a4\ud0ed":3,"\uc2a4\ud3ec\ub97c":3,"\uc2dc\uac04":8,"\uc2dc\uadf8\ub9c8":4,"\uc2dc\uc791\ud558\uace0":1,"\uc2dc\uc791\ud558\uba74":3,"\uc2dc\uc791\ud558\uc5ec":2,"\uc2dc\uc791\ud55c\ub2e4":1,"\uc2dc\uc791\ud55c\ub2e4\uace0":[],"\uc2dc\uc791\ud574\uc11c":1,"\uc2dc\uc810":[0,2],"\uc2dc\uc810\uae4c\uc9c0\ub9cc":2,"\uc2dc\uc810\ub9c8\ub2e4":2,"\uc2dc\uc810\ubcf4\ub2e4":2,"\uc2dc\uc810\ubd80\ud130":2,"\uc2dc\uc810\uc5d0":2,"\uc2dc\uc810\uc5d0\uc11c":1,"\uc2dc\uc810\uc5d0\uc11c\uc758":1,"\uc2dc\uc810\uc758":[1,2],"\uc2dc\uc810\uc778":1,"\uc2dd":[2,3,4],"\uc2dd\ub3c4":4,"\uc2dd\uc5d0\uc11c":4,"\uc2dd\uc740":4,"\uc2dd\uc744":[3,4],"\uc2dd\uc774":[3,4],"\uc2e4\uc218\uac12\uc774\uba70":1,"\uc2e4\uc81c":[2,3,4],"\uc2e4\uc81c\ub85c":2,"\uc2e4\ud604\uac12":[2,3,4],"\uc2e4\ud604\uac12\uc744":4,"\uc2e4\ud604\uac12\uc774":[],"\uc2ec\uce35":[2,4],"\uc2ec\uce35\uac15\ud654\ud559\uc2b5\uc744":8,"\uc2f6\uc73c\uba74":1,"\uc2f6\uc740":1,"\uc2f6\uc744":1,"\uc4f0\ub294":8,"\uc544\ub2c8\uace0":4,"\uc544\ub2c8\uae30":8,"\uc544\ub2c8\ub2e4":[2,8],"\uc544\ub2d0\uae4c":1,"\uc544\ub2d8":4,"\uc544\ub798":[2,3],"\uc544\ubb34\ud2bc":2,"\uc544\uc774\ub514\ub9cc":8,"\uc544\uc774\ub514\uc5b4\ub294":4,"\uc544\uc8fc":[3,8],"\uc544\uc9c1":[1,4],"\uc548":2,"\uc548\uc5d0":[2,3,4],"\uc548\uc758":3,"\uc548\ucabd":3,"\uc549\uae30":1,"\uc54a\uace0":[2,8],"\uc54a\ub294":[1,2],"\uc54a\ub294\ub2e4":1,"\uc54a\uc544\uc11c":1,"\uc54a\uc558\ub2e4":8,"\uc54a\uc740":8,"\uc54a\uc744":3,"\uc54a\uc9c0\ub9cc":2,"\uc54c":[0,1,2,4],"\uc54c\uace0":1,"\uc54c\uace0\ub9ac\uc998\ub4e4\uc740":2,"\uc54c\uace0\ub9ac\uc998\uc758":1,"\uc54c\uace0\ub9ac\uc998\uc774":2,"\uc54c\uae30":2,"\uc54c\ub824\uc8fc\ub294":2,"\uc54c\ub9de\uc740":2,"\uc54c\uba74":[1,2],"\uc54c\uc544\ub450\uba74":2,"\uc54c\uc544\ubcf4\uc558\ub2e4":[2,3],"\uc54c\uc544\ubcf4\uc790":[2,3],"\uc54c\uc544\ubcf8\ub2e4":[2,3],"\uc54c\uc544\ubcfc":[1,3],"\uc54c\uc544\uc57c":1,"\uc54c\uc558\ub2e4":8,"\uc55e\uc11c":2,"\uc55e\uc804\uc5d0":1,"\uc5b4\ub514\uc5d0":1,"\uc5b4\ub518\uac00":8,"\uc5b4\ub5a4":[1,2,3,4],"\uc5b4\ub5a4\uc9c0\ub9cc":8,"\uc5b4\ub5bb\uac8c":[1,2,4],"\uc5b4\ub824\uc6b4":3,"\uc5b4\ub824\uc6b8":1,"\uc5b4\ub835\ub2e4":4,"\uc5b4\uca4c\uba74":2,"\uc5b8\uae09\ud558\uc9c0":1,"\uc5b8\uae09\ud588\uc5c8\ub294\ub370":1,"\uc5bb\uac8c":[0,3],"\uc5bb\ub294\uac00\ub97c":2,"\uc5bb\uc740":2,"\uc5bc\ub9c8\ub098":2,"\uc5c4\uccad\ub098\uac8c":2,"\uc5c5\ub370\uc774\ud2b8":4,"\uc5c5\ub370\uc774\ud2b8\uc2dd\uc774\uae30":4,"\uc5c5\ub370\uc774\ud2b8\ud560":2,"\uc5c6\uae30":[2,3],"\uc5c6\ub2e4\uace0":3,"\uc5c6\uc744":4,"\uc5c6\uc774":[1,2,4,8],"\uc5d0":[1,2,3,4],"\uc5d0\ub294":4,"\uc5d0\ub2e4\uac00":3,"\uc5d0\uc11c":[1,2,3],"\uc5d0\uc11c\uc758":[1,2,3,4],"\uc5d0\uc774\uc804\ud2b8\uac00":[0,1],"\uc5d0\uc774\uc804\ud2b8\ub294":[0,1,2],"\uc5d0\uc774\uc804\ud2b8\uc5d0":1,"\uc5d0\uc774\uc804\ud2b8\uc5d0\uac8c":0,"\uc5d0\uc774\uc804\ud2b8\uc758":[0,1,2],"\uc5d0\ud53c\uc18c\ub4dc\uc5d0\uc11c":4,"\uc5ec\uae30\uae4c\uc9c0\uac00":3,"\uc5ec\uae30\uc11c":[1,2,3],"\uc5ec\uae30\uc5d0":3,"\uc5ec\ub7ec\ubd84\ub4e4\uc5d0\uac8c":8,"\uc5ec\uc12f":1,"\uc5ed\uc2dc":8,"\uc5ed\ud560\uc744":[1,2],"\uc5f0\uad6c\ub418\uc5b4":8,"\uc601\ubb38":8,"\uc608\ub97c":[1,3],"\uc608\uc2dc":3,"\uc608\uc758\ucc28\ub9bc":8,"\uc608\uc815\uc774\ub2e4":[1,2,3],"\uc608\uc815\uc77c\uaec4":2,"\uc624\ub79c":8,"\uc624\ub958\ub97c":8,"\uc624\ub978\ucabd\uc73c\ub85c":1,"\uc628":8,"\uc640":[2,3],"\uc640\ub2ff\uc9c0":[1,3],"\uc644\ubcbd\ud558\uac8c":1,"\uc644\uc804\ud788":4,"\uc67c\ucabd\uc73c\ub85c":1,"\uc694\uc18c":2,"\uc694\uc18c\uac00":4,"\uc694\uc57d\ud558\uc5ec":[],"\uc694\ucee8\ub370":2,"\uc6a9\uc5b4\ub294":4,"\uc6a9\uc5b4\ub97c":8,"\uc6b0\ub9ac\uac00":1,"\uc6b0\ub9ac\ub294":[1,2,3,4],"\uc6b0\ub9ac\uc758":[1,4],"\uc6b0\ubcc0\uc5d0\uc11c":3,"\uc6b0\ubcc0\uc758":3,"\uc6b0\uc120":2,"\uc6b0\uc120\uc740":3,"\uc6d0\uc810\uc5d0\uc11c":1,"\uc6d0\ud65c\ud55c":8,"\uc704":[2,4],"\uc704\uc758":4,"\uc704\uce58\uc640":1,"\uc704\ud558\uc5ec":[1,2,3],"\uc704\ud55c":[0,2,3],"\uc704\ud574":[1,2],"\uc704\ud574\uc11c":1,"\uc704\ud574\uc11c\ub294":[1,2],"\uc704\ud574\uc11c\ub77c\uba74":8,"\uc720\ub3c4\ub41c\ub2e4":3,"\uc720\ub3c4\ud560":3,"\uc720\ub3c4\ud574\ub0bc":4,"\uc720\ub3c4\ud574\ubcf4\uc790":3,"\uc720\ub3c4\ud588\ub358":3,"\uc720\uc2ec\ud788":4,"\uc720\uc6a9\ud558\uac8c":3,"\uc720\uc9c0\ub418\uace0":2,"\uc720\ud55c":3,"\uc720\ud55c\ud558\ub2e4\uace0":3,"\uc73c\ub85c":[1,2,3,4],"\uc740":[1,2,3,4],"\uc744":[0,1,2,3,4],"\uc758":[1,2,3,4,8],"\uc758\ubbf8\uac00":1,"\uc758\ubbf8\ub294":[],"\uc758\ubbf8\uc774\ub2e4":[1,2],"\uc758\ubbf8\ud558\uba70":2,"\uc758\ubbf8\ud55c\ub2e4":[2,8],"\uc758\ubbf8\ud560\uae4c":2,"\uc758\uc0ac":[1,2],"\uc758\ud574":[1,2],"\uc758\ud574\uc11c\ub9cc":1,"\uc774":[0,1,2,3,4,8],"\uc774\uace0":[1,2],"\uc774\ub294":3,"\uc774\ub2e4":[1,3,4],"\uc774\ub3d9":1,"\uc774\ub4dd":2,"\uc774\ub54c":[2,3,4],"\uc774\ub77c\uace0":[0,1,4],"\uc774\ub77c\uace0\ub3c4":4,"\uc774\ub77c\ub294":[2,3],"\uc774\ub780":8,"\uc774\ub97c":[2,3,4],"\uc774\ub984\uc5d0\uc11c":0,"\uc774\uba70":2,"\uc774\uba74":2,"\uc774\ubc88":[2,3,4],"\uc774\uc0c1":[1,3],"\uc774\uc57c\uae30\ub97c":[2,3],"\uc774\uc5b4\uc9c0\ub294":3,"\uc774\uc5d0":3,"\uc774\uc6a9\ud558\uc5ec":4,"\uc774\uc6a9\ud574\uc11c":3,"\uc774\uc720\ub294":[1,2],"\uc774\uc804":4,"\uc774\uc81c":[2,3],"\uc774\ucc98\ub7fc":4,"\uc774\ud574\uac00":3,"\uc774\ud574\ub97c":8,"\uc774\ud574\ud558\uace0":8,"\uc774\ud574\ud558\uae30":3,"\uc774\ud574\ud558\uba74":8,"\uc774\ud574\ud558\uc790\uba74":3,"\uc774\ud574\ud558\uc9c0":8,"\uc774\ud574\ud560":[2,8],"\uc774\ud574\ud574\uc57c\ub9cc":8,"\uc774\ud6c4":2,"\uc774\ud6c4\ubd80\ud130\ub294":2,"\uc774\ud6c4\uc5d0":2,"\uc775\uc219\ud55c":2,"\uc778":[1,2],"\uc778\ud130\ub137":8,"\uc77c\ub828\uc758":1,"\uc77c\ubc18\uc801\uc73c\ub85c":1,"\uc77c\ubc18\uc801\uc778":2,"\uc77c\ubd80":1,"\uc77c\uc815\ud558\ub2e4\uace0":1,"\uc77d\ub2e4\uac00":8,"\uc77d\uc5b4\uc57c":8,"\uc77d\uc73c\uba74":1,"\uc77d\uc74c":1,"\uc785\ub825":[2,8],"\uc785\ubb38\ud560":8,"\uc788\uac8c":[1,3],"\uc788\uace0":2,"\uc788\uae30":[1,2,3,4],"\uc788\ub294":[0,1,2,3,4,8],"\uc788\ub294\ub370":2,"\uc788\ub294\uc9c0\uc774\ub2e4":3,"\uc788\ub2e4":[1,2,3,4,8],"\uc788\ub2e4\uace0":8,"\uc788\ub2e4\ub294":[2,3,4],"\uc788\ub3c4\ub85d":8,"\uc788\uc5b4\uc57c":4,"\uc788\uc5c8\ub294\uc9c0":1,"\uc788\uc73c\ub2c8":8,"\uc788\uc73c\uba74":[2,8],"\uc788\uc744":[1,2,8],"\uc788\uc744\uc9c0":2,"\uc788\uc9c0\ub9cc":[1,2,3],"\uc790":2,"\uc790\uc138\ud788":[1,3],"\uc790\uc5f0\uc5b4":8,"\uc791\uc131\ub41c":8,"\uc791\uc544\uc9c0\ub294":4,"\uc791\uc740":[2,4,8],"\uc798":[1,2,3,4,8],"\uc7a0\uae50":2,"\uc7a5\uc5d0\uc11c":[3,4],"\uc7a5\uc5d0\uc11c\ub294":3,"\uc7a5\uc758":4,"\uc7ac\uadc0\uc801\uc73c\ub85c":3,"\uc7ac\uc57c\uc758":8,"\uc800\uc7a5\ud558\uace0":4,"\uc801\uac8c":0,"\uc801\ub294":4,"\uc801\ub2f9\ud788":1,"\uc801\uc5b4\ub193\uc740":2,"\uc801\uc5b4\ubcf4\uba74":4,"\uc801\uc5b4\ubcf4\uc790":3,"\uc801\uc5b4\ubcfc":3,"\uc801\uc5b4\uc8fc\uba74":[1,4],"\uc801\uc5b4\uc8fc\uc5c8\uc73c\uba70":2,"\uc801\uc5b4\uc900\ub2e4":[1,2,4],"\uc801\uc5b4\uc904":[1,2,3],"\uc801\uc5b4\uc918\ubcf4\uc790":[],"\uc801\uc5b4\uc918\uc57c":[2,3],"\uc801\uc6a9\ud560":[2,3],"\uc801\uc6a9\ud574\ubcf4\uc790":4,"\uc801\uc740":2,"\uc801\uc744":3,"\uc801\uc808\ud558\uc9c0":0,"\uc804\uac1c\ud574":8,"\uc804\ubd80":1,"\uc804\uc5d0":1,"\uc804\uc774":[2,3],"\uc804\uc774\ud560":1,"\uc804\ud1b5\uc801\uc778":8,"\uc808\uc5d0":2,"\uc808\uc5d0\uc11c":[1,2],"\uc808\uc5d0\uc11c\ub294":2,"\uc810\uc740":3,"\uc810\uc810":4,"\uc810\ud504\uac00":1,"\uc815\ub9ac\ud558\uc790\uba74":3,"\uc815\ub9d0":4,"\uc815\uc758":3,"\uc815\uc758\uac00":2,"\uc815\uc758\ub294":[1,3],"\uc815\uc758\ub418\uae30":2,"\uc815\uc758\ub418\ub294":2,"\uc815\uc758\ub418\ub294\uc9c0":1,"\uc815\uc758\ub418\uba70":1,"\uc815\uc758\ub418\uc5b4":[1,2],"\uc815\uc758\ub41c":2,"\uc815\uc758\ub41c\ub2e4":[2,4],"\uc815\uc758\ub41c\ub2e4\uace0":1,"\uc815\uc758\ub97c":[1,2,3],"\uc815\uc758\uc0c1":2,"\uc815\uc758\uc5d0":2,"\uc815\uc758\uc5d0\uc11c":[],"\uc815\uc758\uc5ed\uc774":1,"\uc815\uc758\uc778":[2,3],"\uc815\uc758\ud558\ub294":2,"\uc815\uc758\ud558\uc5ec":1,"\uc815\uc758\ud55c\ub2e4\uba74":1,"\uc815\uc758\ud560":1,"\uc815\uc758\ud574\ubcf4\uc790":1,"\uc815\ucc45":[3,4],"\uc815\ucc45\uacfc":2,"\uc815\ucc45\uc5d0":2,"\uc815\ucc45\uc740":2,"\uc815\ucc45\uc744":[2,3],"\uc815\ucc45\uc758":[2,3],"\uc815\ucc45\uc774":2,"\uc815\ucc45\uc774\ub2e4":2,"\uc815\ud574\uc838":[1,2],"\uc815\ud574\uc838\uc788\ub2e4\uace0":1,"\uc815\ud574\uc9c4":[0,1,2],"\uc81c\uc5b4\uc758":1,"\uc81c\uc5b4\ud55c\ub2e4":2,"\uc81c\uc678\ud558\uace0":4,"\uc81c\uc77c":8,"\uc81c\ud55c\ub418\uc5b4":1,"\uc870\uac74":2,"\uc870\uac74\ub9cc\ud07c":0,"\uc870\uac74\ubd80":[1,2],"\uc870\uac74\ubd80\uc5d0":[2,3],"\uc870\uac74\ubd80\uc758":2,"\uc870\uac74\uc5d0":2,"\uc870\uae08":[1,2,3],"\uc870\uae08\ub9cc":2,"\uc870\uae08\uc529":1,"\uc870\uc791\ud574\ubcf4\uc790":4,"\uc885\ub8cc":2,"\uc88b\uaca0\ub2e4":8,"\uc88b\uaca0\uc9c0\ub9cc":8,"\uc88b\uace0":[1,2,3],"\uc88b\ub2e4\uace0":2,"\uc88b\uc740":[0,2,3,8],"\uc88b\uc744":4,"\uc88c\ubcc0\uacfc":3,"\uc8fc\ub294":[1,2],"\uc8fc\ub85c":[1,4],"\uc8fc\uc5b4\uc84c\ub2e4\ub294":2,"\uc8fc\uc5b4\uc84c\uc744":1,"\uc8fc\uc5b4\uc9c4":2,"\uc8fc\uc758\ud560":[],"\uc8fc\uc778\uacf5\uc744":0,"\uc8fc\uc800\ud558\uc9c0":8,"\uc900\ube44\ub41c":2,"\uc904":8,"\uc904\uc138\uc6e0\uc744":2,"\uc904\uc784\ub9d0":8,"\uc911":[2,3],"\uc911\ubcf5\ub418\uae30":4,"\uc911\uc5d0\uc11c":8,"\uc911\uc694\ud55c":[2,3],"\uc989":[1,2],"\uc99d\uba85\ub418\uc9c0\ub9cc":4,"\uc99d\uba85\uc740":2,"\uc99d\uba85\uc744":4,"\uc99d\uba85\ud558\ub294":[3,4],"\uc9c0\uae08\uae4c\uc9c0":[2,8],"\uc9c0\uae08\ubd80\ud130":2,"\uc9c0\ub09c":3,"\uc9c0\uce68\uc11c\uc774\ub2e4":2,"\uc9c0\ud45c\uc774\ub2e4":3,"\uc9c1\uad00\uc131\uc744":1,"\uc9c1\uad00\uc801\uc778":3,"\uc9c1\uc5ed\ud558\uba74":1,"\uc9c1\uc811":3,"\uc9c1\uc811\uc801\uc73c\ub85c":1,"\uc9c4\ud589\ub418\ub294":2,"\uc9c4\ud589\ub418\uc5c8\uc5b4\ub3c4":2,"\uc9c4\ud589\ub420":2,"\uc9c8\ubb38\uc5d0":2,"\uc9c8\ubb38\uc774":8,"\uc9d1\uc5b4":3,"\uc9d1\ud569\uc774\ub2e4":1,"\uc9d1\ud569\uc774\ub77c\uace0":1,"\uc9f8":1,"\ucb49":2,"\ucc28\uadfc":2,"\ucc28\ub840\uc774\ub2e4":3,"\ucc28\uc774\uac00":3,"\ucc3e\uc544\ub0b4\ub294":2,"\ucc3e\uc544\uc11c":2,"\ucc3e\uc73c\uba74":2,"\ucc45\uc5d0\uc11c\ub294":[1,2],"\ucc45\uc740":8,"\ucc45\uc744":8,"\ucc45\uc758":8,"\ucc98\ub9ac":8,"\ucc98\uc74c\ubd80\ud130":8,"\ucc98\uc74c\uc5d0\ub294":1,"\ucca8\uc790\ub85c":2,"\uccab":[1,3],"\ucd08\uae30":[1,2],"\ucd08\ubcf4":8,"\ucd1d":1,"\ucd1d\ud569\uc744":[0,2],"\ucd5c\ub300":1,"\ucd5c\ub300\uac12\uc774":1,"\ucd5c\ub300\ub85c":2,"\ucd5c\ub300\ud55c":2,"\ucd5c\ub300\ud654\ud558\ub294":[0,2],"\ucd5c\uc2e0":8,"\ucd5c\uc801\uc758":2,"\ucd5c\uc801\ud654\ud558\uc5ec":2,"\ucd5c\uc885\uc801\uc73c\ub85c":[],"\ucd94\uac00\ub418\uc5c8\uc744":4,"\ucd94\uac00\ub41c":3,"\ucd94\uac00\ub420":4,"\ucd94\uac00\uc801\uc73c\ub85c":1,"\ucd94\uc0c1\uc801\uc778":1,"\ucd94\uc815":3,"\ucd94\uc815\uac12":4,"\ucd94\uc815\ud558\uac70\ub098":2,"\ucd94\uc815\ud558\uac8c":2,"\ucd94\uc815\ud558\uae30":2,"\ucd94\uc815\ud55c\ub2e4":4,"\ucd94\uc815\ud560":4,"\ucd9c\ub825\ud558\ub294":2,"\ucd9c\ud310\uc744":8,"\ucda9\ubd84\ud788":4,"\ucde8\ud558\uace0":2,"\ucde8\ud558\uae30":1,"\ucde8\ud558\ub3c4\ub85d":2,"\ucde8\ud558\uba70":0,"\ucde8\ud558\uba74":2,"\ucde8\ud558\uc5ec":1,"\ucde8\ud55c":[1,2],"\ucde8\ud55c\ub2e4":0,"\ucde8\ud560":[1,2],"\ucde8\ud574\uc11c":2,"\ucde8\ud574\uc57c":2,"\ucde8\ud588\ub2e4\uba74":0,"\ucde8\ud588\uc744":[1,2],"\uce21\uc815\ub420":2,"\uce21\uc815\ud560":2,"\ucef4\ud4e8\ud130":8,"\ucf00\uc8fc\uc5bc\ud558\uac8c":8,"\ud06c\uac8c":[2,4],"\ud06c\uae30\uac00":3,"\ud06c\ub0e5":4,"\ud06c\ub2e4":2,"\ud070":2,"\ud074":2,"\ud0a4":3,"\ud0a4\ub97c":3,"\ud0c0\uc6b0\ub77c\uace0":1,"\ud0c4\ub3c4":1,"\ud140\ubd80\ud130":3,"\ud1b5\uacc4":4,"\ud1b5\uc81c\ud558\uace0\uc790":0,"\ud1b5\ud574":[2,4],"\ud2b9\ubcc4\ud55c":2,"\ud2b9\uc815":[1,2],"\ud2b9\ud788":4,"\ud2c0\uc774":1,"\ud2c0\uc774\ub2e4":1,"\ud30c\ub77c\ubbf8\ud130\ub97c":2,"\ud398\uc774\uc9c0\ub9c8\ub2e4":8,"\ud3b8\uc758\uc131\uc744":3,"\ud3bc\uce5c":8,"\ud3c9\uac00\ud558\uae30":2,"\ud3c9\uac00\ud558\uae30\uc5d4":2,"\ud3c9\uac00\ud560":[2,3],"\ud3c9\uade0":[3,4],"\ud3c9\uade0\uc744":[3,4],"\ud3ec\uc2a4\ud305\uc5d0":4,"\ud3ec\ud568\uc2dc\ud0a4\uc9c0":4,"\ud3ec\ud568\ud558\uc5ec":3,"\ud45c\uae30\uc5d0\ub294":2,"\ud45c\uae30\uc758":3,"\ud45c\uae30\ud558\uace0":[],"\ud45c\uae30\ud558\ub824\uace0":2,"\ud45c\uae30\ud558\uc600\ub2e4":[],"\ud45c\uae30\ud558\uc790":1,"\ud45c\uae30\ud574\uc900\ub2e4":[1,2],"\ud45c\ubcf8":4,"\ud45c\ud604\ub418\uc5b4":4,"\ud45c\ud604\ub41c":3,"\ud45c\ud604\ud55c":1,"\ud45c\ud604\ud560":1,"\ud45c\ud604\ud574\uc900\ub2e4":3,"\ud45c\ud604\ud588\ub2e4":1,"\ud480\uc5b4\ub0b4\uace0\uc790":0,"\ud544\uc694":[1,4],"\ud544\uc694\ud558\ub2e4":[1,4],"\ud544\uc694\ud558\uc9c0":8,"\ud544\uc694\ud55c":2,"\ud544\uc790\ub294":[4,8],"\ud558\uac8c":3,"\ud558\ub098":2,"\ud558\ub098\ub294":2,"\ud558\ub098\ub85c":[2,4],"\ud558\ub098\ub97c":3,"\ud558\ub098\uc758":2,"\ud558\ub098\uc774\ub2e4":2,"\ud558\ub098\uc778":[2,3],"\ud558\ub294":[0,1,2,8],"\ud558\ub2e4":3,"\ud558\ub354\ub77c\ub3c4":8,"\ud558\uc5ec":0,"\ud558\uc790":2,"\ud558\uc790\uba74":3,"\ud558\uc9c0":[],"\ud558\uc9c0\ub9cc":[1,2,3,4],"\ud559\ubb38\uc801\uc73c\ub85c":8,"\ud559\uc0dd\ub4e4\uc758":3,"\ud559\uc220\uc801\uc73c\ub85c\ub294":1,"\ud559\uc2b5":2,"\ud559\uc2b5\uc2dc\ud0a8\ub2e4":3,"\ud559\uc2b5\uc2dc\ud0ac":2,"\ud559\uc2b5\uc5d0":2,"\ud559\uc2b5\uc5d0\ub3c4":4,"\ud559\uc2b5\uc5d0\uc11c":2,"\ud559\uc2b5\uc744":2,"\ud559\uc2b5\ud560":2,"\ud55c":[1,2,8],"\ud55c\ub2e4":[0,1,2,3,4,8],"\ud55c\ub2e4\ub294":3,"\ud55c\ubc88":[2,3],"\ud55c\ubc88\uc5d0":1,"\ud55c\ud3b8":[2,3],"\ud560":[2,3],"\ud560\uc778\ub41c":[2,3],"\ud560\uc778\ub960":2,"\ud560\uc778\ub960\uc5d0":1,"\ud560\uc9c0":2,"\ud568\uc218":3,"\ud568\uc218\uac00":[2,4],"\ud568\uc218\uac12\ub9c8\ub2e4":1,"\ud568\uc218\uac12\uc774":2,"\ud568\uc218\ub294":[1,2,3],"\ud568\uc218\ub3c4":3,"\ud568\uc218\ub77c\uace0":[1,2],"\ud568\uc218\ub77c\ub294":2,"\ud568\uc218\ub77c\uba74":3,"\ud568\uc218\ub85c":[1,2],"\ud568\uc218\ub85c\uc11c":2,"\ud568\uc218\ub97c":[2,3,4],"\ud568\uc218\ub9cc":2,"\ud568\uc218\uc5d0":[2,3],"\ud568\uc218\uc5d0\ub294":2,"\ud568\uc218\uc5d0\uc11c":2,"\ud568\uc218\uc600\ub2e4\uba74":2,"\ud568\uc218\uc640":[2,3],"\ud568\uc218\uc758":[1,2],"\ud568\uc218\uc774\uace0":1,"\ud568\uc218\uc774\ub2e4":[1,2],"\ud568\uc218\uc774\uba74":3,"\ud569":2,"\ud569\ub9ac\uc801\uc778":1,"\ud569\uc758":3,"\ud56d\uc0c1":4,"\ud574\uacb0\ud558\uae30":[0,1],"\ud574\uacb0\ud558\ub294":1,"\ud574\ub2f9":[1,2,3,8],"\ud574\ub2f9\ud558\uace0":3,"\ud574\ub2f9\ud55c\ub2e4":[3,4],"\ud574\ubcf4\uc790":2,"\ud574\ubcf4\uc790\uba74":3,"\ud574\ubcfc":8,"\ud574\uc11d\ud560":[1,2],"\ud589\ub3d9":0,"\ud589\ub3d9\uacf5\uac04":1,"\ud589\ub3d9\uae4c\uc9c0\uac00":1,"\ud589\ub3d9\ub4e4\uc758":1,"\ud589\ub3d9\uc5d0":[1,2],"\ud589\ub3d9\uc73c\ub85c":1,"\ud589\ub3d9\uc740":[1,2],"\ud589\ub3d9\uc744":[0,1,2],"\ud589\ub3d9\uc758":[1,2],"\ud589\ub3d9\uc774":[2,3],"\ud604\ub300\uc758":8,"\ud604\uc7ac":[1,2,4],"\ud604\uc7ac\uc758":1,"\ud615\ud0dc\uac00":[],"\ud615\ud0dc\ub85c":4,"\ud615\ud0dc\uc640":8,"\ud615\ud0dc\uc774\ub2e4":3,"\ud615\ud0dc\uc778":3,"\ud655\ub960":[2,3,4],"\ud655\ub960\uacfc":[3,4],"\ud655\ub960\ub85c":2,"\ud655\ub960\ub860":4,"\ud655\ub960\ub860\uc744":4,"\ud655\ub960\ub860\uc801\uc778":2,"\ud655\ub960\ub9cc":2,"\ud655\ub960\uc5d0":2,"\ud655\ub960\uc744":[1,2],"\ud655\ub960\uc774":[1,2],"\ud655\ub960\uc801":2,"\ud658\uacbd\uacfc":2,"\ud658\uacbd\uc5d0\uc11c":2,"\ud658\uacbd\uc740":[0,1,2],"\ud658\uacbd\uc744":2,"\ud658\uacbd\uc758":[0,1,2],"\ud658\uacbd\uc774":1,"\ud68c\uc804\ud558\ub294":1,"\ud68c\uc804\ud560":1,"\ud69f\uc218\ub098":0,"\ud6c4":2,"approximation\uc744":4,"approximation\uc758":4,"approximation\uc774":4,"art\uc758":8,"carlo\uc640":4,"case":2,"equation\uc774\ub77c\uace0":3,"expectation\ub97c":3,"expectation\uc740":3,"expectation\uc744":3,"function":[],"learning\uc758":4,"mdp\uac00":1,"mdp\ub294":1,"mdp\ub85c":[1,2],"mdp\ub97c":1,"process\ub77c\uace0":1,"property\ub97c":1,"return":4,"return\ub3c4":3,"return\uc740":2,"return\uc744":[2,3],"return\uc758":2,"return\uc774":2,"return\uc774\ub77c\uace0":3,"return\uc778":3,"stochastic\ud558\uac8c":1,"stochastic\ud55c":[],"stochastic\ud560":[],"trajectory\ub77c\uace0":1,"trajectory\ub97c":[1,2],"trajectory\uc5d0\uc11c":1,A:[1,2,3],In:7,_:[2,3,4],a_0:[1,2],a_1:[1,2],a_2:[],a_:[1,2,3],a_k:2,a_t:[1,2,3],abbeel:7,action:0,actor:[7,8],advantag:[],agent:0,align:[],all:2,alpha:4,alpha_:4,andrea:7,approx:4,approxim:3,argmax:2,aurick:7,begin:[2,3,4],bellman:3,bellman_equ:[],bound:1,br:[],cdot:[2,3],circ:1,come:[5,6],confer:7,critic:[7,8],cummul:2,decis:0,deep:7,determinist:2,determinit:1,discount:2,dy:7,e:[2,3,4],editor:7,end:[2,3,4],entropi:7,environ:0,eq:[],equat:[],estim:[3,4],expect:[],f:4,foral:[2,3],frac:4,g_:[3,4],g_t:[2,3,4],gamma:[1,2,3,4],ge:2,gnn:8,haarnoja18b:7,haarnoja:7,html:7,http:7,hzal18:[],i:4,increment:4,intern:7,intract:3,iter:4,j:2,jennif:7,k:[2,4],kraus:7,law:[],ldot:[1,2,3],learn:7,left:[1,2,3,4],leftarrow:4,levin:7,limits_:[2,3,4],machin:7,mai:7,make:0,mathbb:[1,2,3,4],mathcal:[1,2,3],matrix:[3,4],maximum:7,mean:4,mlr:7,n:[3,4],note:[],observ:4,off:7,operatornam:2,optim:2,order:2,otherwis:2,p:[1,2,3],pi:[2,3,4],pi_:2,pieter:7,pmlr:7,polici:7,pr:[1,2],press:7,problem:0,proceed:7,q:[2,3],q_:3,quad:[2,3],r:[1,2,3],r_0:[1,2],r_1:[1,2],r_2:1,r_3:1,r_:[2,3,4],r_t:[1,2,3,4],random:[2,3,4],realiz:[2,4],reinforc:7,research:7,reward:0,rho:[],rho_0:[1,2],right:[1,2,3,4],rightarrow:[1,2],s:[1,2,3,4],s_0:[1,2],s_1:[1,2],s_2:[],s_3:[],s_:[1,2,3,4],s_k:2,s_n:4,s_t:[1,2,3,4],sequenti:0,sergei:7,seri:7,sim:[1,2,3],soft:[7,8],soon:[5,6],sota:8,state:[0,8],state_bellman_equ:[],state_value_fnct:[],state_value_funct:[],step:3,sthochast:4,stochast:[2,3,7],sum:[2,3,4],sum_:3,t:[1,2,3,4],tau:[1,2],td:4,termin:2,text:[1,2],theta:[2,3,4],time:[1,2],titl:7,total:[],trajectori:1,tuoma:7,url:7,v80:7,v:[2,3,4],v_:[3,4],v_n:4,valu:[],variabl:[2,3,4],volum:7,x:[3,4],x_:4,x_i:4,x_n:4,xi:4,y:3,zeta:[],zhou:7},titles:["<span class=\"section-number\">1. </span>\uc21c\ucc28\uc801 \uc758\uc0ac \uacb0\uc815 \ubb38\uc81c, \uc5d0\uc774\uc804\ud2b8, \ud658\uacbd","<span class=\"section-number\">2. </span>Markov Decision Process (MDP)","<span class=\"section-number\">3. </span>\uc815\ucc45, Return, \uac00\uce58 \ud568\uc218","<span class=\"section-number\">4. </span>\ubca8\ub9cc \ubc29\uc815\uc2dd: \uac00\uce58 \ud568\uc218\uc758 \uc7ac\uadc0\uc801 \uc131\uc9c8","<span class=\"section-number\">5. </span>\uac00\uce58 \ud568\uc218 \uadfc\uc0ac\ud558\uae30: Stochastic approximation","<span class=\"section-number\">6. </span>Policy Gradient Theorem","<span class=\"section-number\">6. </span>REINFORCE","\ucc38\uace0\ubb38\ud5cc","\uc2ec\uce35\uac15\ud654\ud559\uc2b5 (Deep Reinforcement Learnings)"],titleterms:{"\uac00\uce58":[2,3,4],"\uacb0\uc815":0,"\uacf5\uac04":1,"\uacfc":1,"\uadfc\uc0ac\ud558\uae30":4,"\ubb38\uc81c":0,"\ubc29\uc815\uc2dd":3,"\ubca8\ub9cc":3,"\ubcf4\uc0c1":1,"\ubd84\ud3ec":1,"\uc0c1\ud0dc":[1,2,3],"\uc131\uc9c8":3,"\uc21c\ucc28\uc801":0,"\uc2ec\uce35\uac15\ud654\ud559\uc2b5":8,"\uc5d0\uc774\uc804\ud2b8":0,"\uc758\uc0ac":0,"\uc7ac\uadc0\uc801":3,"\uc804\uc774":1,"\uc815\ucc45":2,"\ucc38\uace0\ubb38\ud5cc":7,"\ud45c\ud604":3,"\ud560\uc778\ub960":1,"\ud568\uc218":[1,2,4],"\ud568\uc218\uc758":3,"\ud589\ub3d9":[1,2,3],"\ud655\ub960":1,"\ud658\uacbd":0,"function":[1,2],"return":2,"return\uc758":3,"trajectory\uc640":1,action:[1,2],advantag:2,approxim:4,carlo:4,decis:1,deep:8,definit:[],differ:4,discount:1,distribut:1,evalu:4,expect:3,factor:1,gradient:5,law:3,learn:8,markov:1,mdp:1,mont:4,polici:[2,5],probabl:1,process:1,reinforc:[6,8],reward:1,space:1,state:[1,2],stochast:4,tempor:4,theorem:5,total:3,transit:1,valu:2}})