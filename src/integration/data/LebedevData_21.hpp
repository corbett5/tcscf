#pragma once

// Lebedev quadrature points of order 21.
// The ordering of the data is { polar, azimuthal, weight }.
// Angles are in radians with the polar angle between 0 and pi
// and the azimuthal angle between -pi and pi.

static constexpr long double Lebedev_21[ 170 ][ 3 ] = {
  {  1.5707963267948966192313217L,  0.0000000000000000000000000L,  0.0696785509053957951444800L },
  {  1.5707963267948966192313217L,  3.1415926535897932384626434L,  0.0696785509053957951444800L },
  {  1.5707963267948966192313217L,  1.5707963267948966192313217L,  0.0696785509053957951444800L },
  {  1.5707963267948966192313217L, -1.5707963267948966192313217L,  0.0696785509053957951444800L },
  {  0.0000000000000000000000000L,  1.5707963267948966192313217L,  0.0696785509053957951444800L },
  {  3.1415926535897932384626434L,  1.5707963267948966192313217L,  0.0696785509053957951444800L },
  {  0.7853981633974483096156608L,  1.5707963267948966192313217L,  0.0762946177193559139870457L },
  {  2.3561944901923449288469825L,  1.5707963267948966192313217L,  0.0762946177193559139870457L },
  {  0.7853981633974483096156608L, -1.5707963267948966192313217L,  0.0762946177193559139870457L },
  {  2.3561944901923449288469825L, -1.5707963267948966192313217L,  0.0762946177193559139870457L },
  {  0.7853981633974483096156608L,  0.0000000000000000000000000L,  0.0762946177193559139870457L },
  {  2.3561944901923449288469825L,  0.0000000000000000000000000L,  0.0762946177193559139870457L },
  {  0.7853981633974483096156608L,  3.1415926535897932384626434L,  0.0762946177193559139870457L },
  {  2.3561944901923449288469825L,  3.1415926535897932384626434L,  0.0762946177193559139870457L },
  {  1.5707963267948966192313217L,  0.7853981633974483096156608L,  0.0762946177193559139870457L },
  {  1.5707963267948966192313217L, -0.7853981633974483096156608L,  0.0762946177193559139870457L },
  {  1.5707963267948966192313217L,  2.3561944901923449288469825L,  0.0762946177193559139870457L },
  {  1.5707963267948966192313217L, -2.3561944901923449288469825L,  0.0762946177193559139870457L },
  {  0.9553166181245092836682241L,  0.7853981633974483096156608L,  0.0802196230855248448130348L },
  {  2.1862760354652839547944192L,  0.7853981633974483096156608L,  0.0802196230855248448130348L },
  {  0.9553166181245092836682241L, -0.7853981633974483096156608L,  0.0802196230855248448130348L },
  {  2.1862760354652839547944192L, -0.7853981633974483096156608L,  0.0802196230855248448130348L },
  {  0.9553166181245092836682241L,  2.3561944901923449288469825L,  0.0802196230855248448130348L },
  {  2.1862760354652839547944192L,  2.3561944901923449288469825L,  0.0802196230855248448130348L },
  {  0.9553166181245092836682241L, -2.3561944901923449288469825L,  0.0802196230855248448130348L },
  {  2.1862760354652839547944192L, -2.3561944901923449288469825L,  0.0802196230855248448130348L },
  {  0.3691272501334426538238156L,  0.7853981633974483096156608L,  0.0651363694655105462681158L },
  {  2.7724654034563508289849231L,  0.7853981633974483096156608L,  0.0651363694655105462681158L },
  {  0.3691272501334426538238156L, -0.7853981633974483096156608L,  0.0651363694655105462681158L },
  {  2.7724654034563508289849231L, -0.7853981633974483096156608L,  0.0651363694655105462681158L },
  {  0.3691272501334426538238156L,  2.3561944901923449288469825L,  0.0651363694655105462681158L },
  {  2.7724654034563508289849231L,  2.3561944901923449288469825L,  0.0651363694655105462681158L },
  {  0.3691272501334426538238156L, -2.3561944901923449288469825L,  0.0651363694655105462681158L },
  {  2.7724654034563508289849231L, -2.3561944901923449288469825L,  0.0651363694655105462681158L },
  {  1.3128190766509796168506620L,  1.3037777890647554696661804L,  0.0651363694655105462681158L },
  {  1.3128190766509796168506620L, -1.3037777890647554696661804L,  0.0651363694655105462681158L },
  {  1.8287735769388138659580767L,  1.3037777890647554696661804L,  0.0651363694655105462681158L },
  {  1.8287735769388138659580767L, -1.3037777890647554696661804L,  0.0651363694655105462681158L },
  {  1.3128190766509796168506620L,  1.8378148645250380305958508L,  0.0651363694655105462681158L },
  {  1.3128190766509796168506620L, -1.8378148645250380305958508L,  0.0651363694655105462681158L },
  {  1.8287735769388138659580767L,  1.8378148645250380305958508L,  0.0651363694655105462681158L },
  {  1.8287735769388138659580767L, -1.8378148645250380305958508L,  0.0651363694655105462681158L },
  {  1.3128190766509796168506620L,  0.2670185377301412717381890L,  0.0651363694655105462681158L },
  {  1.3128190766509796168506620L,  2.8745741158596520888975021L,  0.0651363694655105462681158L },
  {  1.8287735769388138659580767L,  0.2670185377301412717381890L,  0.0651363694655105462681158L },
  {  1.8287735769388138659580767L,  2.8745741158596520888975021L,  0.0651363694655105462681158L },
  {  1.3128190766509796168506620L, -0.2670185377301412717381890L,  0.0651363694655105462681158L },
  {  1.3128190766509796168506620L, -2.8745741158596520888975021L,  0.0651363694655105462681158L },
  {  1.8287735769388138659580767L, -0.2670185377301412717381890L,  0.0651363694655105462681158L },
  {  1.8287735769388138659580767L, -2.8745741158596520888975021L,  0.0651363694655105462681158L },
  {  1.2652716500810328145954285L,  0.7853981633974483096156608L,  0.0793934374525339964304691L },
  {  1.8763210035087604238672149L,  0.7853981633974483096156608L,  0.0793934374525339964304691L },
  {  1.2652716500810328145954285L, -0.7853981633974483096156608L,  0.0793934374525339964304691L },
  {  1.8763210035087604238672149L, -0.7853981633974483096156608L,  0.0793934374525339964304691L },
  {  1.2652716500810328145954285L,  2.3561944901923449288469825L,  0.0793934374525339964304691L },
  {  1.8763210035087604238672149L,  2.3561944901923449288469825L,  0.0793934374525339964304691L },
  {  1.2652716500810328145954285L, -2.3561944901923449288469825L,  0.0793934374525339964304691L },
  {  1.8763210035087604238672149L, -2.3561944901923449288469825L,  0.0793934374525339964304691L },
  {  0.8306985059283249230175643L,  0.4195583796041280431740930L,  0.0793934374525339964304691L },
  {  0.8306985059283249230175643L, -0.4195583796041280431740930L,  0.0793934374525339964304691L },
  {  2.3108941476614684376181267L,  0.4195583796041280431740930L,  0.0793934374525339964304691L },
  {  2.3108941476614684376181267L, -0.4195583796041280431740930L,  0.0793934374525339964304691L },
  {  0.8306985059283249230175643L,  2.7220342739856653174615981L,  0.0793934374525339964304691L },
  {  0.8306985059283249230175643L, -2.7220342739856653174615981L,  0.0793934374525339964304691L },
  {  2.3108941476614684376181267L,  2.7220342739856653174615981L,  0.0793934374525339964304691L },
  {  2.3108941476614684376181267L, -2.7220342739856653174615981L,  0.0793934374525339964304691L },
  {  0.8306985059283249230175643L,  1.1512379471907686982302764L,  0.0793934374525339964304691L },
  {  0.8306985059283249230175643L,  1.9903547063990245402323670L,  0.0793934374525339964304691L },
  {  2.3108941476614684376181267L,  1.1512379471907686982302764L,  0.0793934374525339964304691L },
  {  2.3108941476614684376181267L,  1.9903547063990245402323670L,  0.0793934374525339964304691L },
  {  0.8306985059283249230175643L, -1.1512379471907686982302764L,  0.0793934374525339964304691L },
  {  0.8306985059283249230175643L, -1.9903547063990245402323670L,  0.0793934374525339964304691L },
  {  2.3108941476614684376181267L, -1.1512379471907686982302764L,  0.0793934374525339964304691L },
  {  2.3108941476614684376181267L, -1.9903547063990245402323670L,  0.0793934374525339964304691L },
  {  0.6570531545351615610580631L,  0.7853981633974483096156608L,  0.0779324837307526681107191L },
  {  2.4845394990546317995776279L,  0.7853981633974483096156608L,  0.0779324837307526681107191L },
  {  0.6570531545351615610580631L, -0.7853981633974483096156608L,  0.0779324837307526681107191L },
  {  2.4845394990546317995776279L, -0.7853981633974483096156608L,  0.0779324837307526681107191L },
  {  0.6570531545351615610580631L,  2.3561944901923449288469825L,  0.0779324837307526681107191L },
  {  2.4845394990546317995776279L,  2.3561944901923449288469825L,  0.0779324837307526681107191L },
  {  0.6570531545351615610580631L, -2.3561944901923449288469825L,  0.0779324837307526681107191L },
  {  2.4845394990546317995776279L, -2.3561944901923449288469825L,  0.0779324837307526681107191L },
  {  1.1242078977252877858085010L,  1.0714470911969618695989524L,  0.0779324837307526681107191L },
  {  1.1242078977252877858085010L, -1.0714470911969618695989524L,  0.0779324837307526681107191L },
  {  2.0173847558645054526541424L,  1.0714470911969618695989524L,  0.0779324837307526681107191L },
  {  2.0173847558645054526541424L, -1.0714470911969618695989524L,  0.0779324837307526681107191L },
  {  1.1242078977252877858085010L,  2.0701455623928316132097863L,  0.0779324837307526681107191L },
  {  1.1242078977252877858085010L, -2.0701455623928316132097863L,  0.0779324837307526681107191L },
  {  2.0173847558645054526541424L,  2.0701455623928316132097863L,  0.0779324837307526681107191L },
  {  2.0173847558645054526541424L, -2.0701455623928316132097863L,  0.0779324837307526681107191L },
  {  1.1242078977252877858085010L,  0.4993492355979348718054169L,  0.0779324837307526681107191L },
  {  1.1242078977252877858085010L,  2.6422434179918587506296619L,  0.0779324837307526681107191L },
  {  2.0173847558645054526541424L,  0.4993492355979348718054169L,  0.0779324837307526681107191L },
  {  2.0173847558645054526541424L,  2.6422434179918587506296619L,  0.0779324837307526681107191L },
  {  1.1242078977252877858085010L, -0.4993492355979348718054169L,  0.0779324837307526681107191L },
  {  1.1242078977252877858085010L, -2.6422434179918587506296619L,  0.0779324837307526681107191L },
  {  2.0173847558645054526541424L, -0.4993492355979348718054169L,  0.0779324837307526681107191L },
  {  2.0173847558645054526541424L, -2.6422434179918587506296619L,  0.0779324837307526681107191L },
  {  1.5707963267948966192313217L,  1.3063310886865492339879152L,  0.0688278136856173229324208L },
  {  1.5707963267948966192313217L, -1.3063310886865492339879152L,  0.0688278136856173229324208L },
  {  1.5707963267948966192313217L,  1.8352615649032440044747282L,  0.0688278136856173229324208L },
  {  1.5707963267948966192313217L, -1.8352615649032440044747282L,  0.0688278136856173229324208L },
  {  1.5707963267948966192313217L,  0.2644652381083472805236514L,  0.0688278136856173229324208L },
  {  1.5707963267948966192313217L, -0.2644652381083472805236514L,  0.0688278136856173229324208L },
  {  1.5707963267948966192313217L,  2.8771274154814461150186247L,  0.0688278136856173229324208L },
  {  1.5707963267948966192313217L, -2.8771274154814461150186247L,  0.0688278136856173229324208L },
  {  0.2644652381083472805236514L,  0.0000000000000000000000000L,  0.0688278136856173229324208L },
  {  2.8771274154814461150186247L,  0.0000000000000000000000000L,  0.0688278136856173229324208L },
  {  0.2644652381083472805236514L,  3.1415926535897932384626434L,  0.0688278136856173229324208L },
  {  2.8771274154814461150186247L,  3.1415926535897932384626434L,  0.0688278136856173229324208L },
  {  1.3063310886865492339879152L,  0.0000000000000000000000000L,  0.0688278136856173229324208L },
  {  1.8352615649032440044747282L,  0.0000000000000000000000000L,  0.0688278136856173229324208L },
  {  1.3063310886865492339879152L,  3.1415926535897932384626434L,  0.0688278136856173229324208L },
  {  1.8352615649032440044747282L,  3.1415926535897932384626434L,  0.0688278136856173229324208L },
  {  0.2644652381083472805236514L,  1.5707963267948966192313217L,  0.0688278136856173229324208L },
  {  2.8771274154814461150186247L,  1.5707963267948966192313217L,  0.0688278136856173229324208L },
  {  0.2644652381083472805236514L, -1.5707963267948966192313217L,  0.0688278136856173229324208L },
  {  2.8771274154814461150186247L, -1.5707963267948966192313217L,  0.0688278136856173229324208L },
  {  1.3063310886865492339879152L,  1.5707963267948966192313217L,  0.0688278136856173229324208L },
  {  1.8352615649032440044747282L,  1.5707963267948966192313217L,  0.0688278136856173229324208L },
  {  1.3063310886865492339879152L, -1.5707963267948966192313217L,  0.0688278136856173229324208L },
  {  1.8352615649032440044747282L, -1.5707963267948966192313217L,  0.0688278136856173229324208L },
  {  0.5463708680698394719662660L,  0.2821463914363193779918707L,  0.0750009251580063385127038L },
  {  2.5952217855199542551885679L,  0.2821463914363193779918707L,  0.0750009251580063385127038L },
  {  0.5463708680698394719662660L, -0.2821463914363193779918707L,  0.0750009251580063385127038L },
  {  2.5952217855199542551885679L, -0.2821463914363193779918707L,  0.0750009251580063385127038L },
  {  0.5463708680698394719662660L,  2.8594462621534737382977251L,  0.0750009251580063385127038L },
  {  2.5952217855199542551885679L,  2.8594462621534737382977251L,  0.0750009251580063385127038L },
  {  0.5463708680698394719662660L, -2.8594462621534737382977251L,  0.0750009251580063385127038L },
  {  2.5952217855199542551885679L, -2.8594462621534737382977251L,  0.0750009251580063385127038L },
  {  1.4256238701470745341917051L,  1.0421665902718458391683995L,  0.0750009251580063385127038L },
  {  1.7159687834427189660703261L,  1.0421665902718458391683995L,  0.0750009251580063385127038L },
  {  1.4256238701470745341917051L, -1.0421665902718458391683995L,  0.0750009251580063385127038L },
  {  1.7159687834427189660703261L, -1.0421665902718458391683995L,  0.0750009251580063385127038L },
  {  1.4256238701470745341917051L,  2.0994260633179475214672915L,  0.0750009251580063385127038L },
  {  1.7159687834427189660703261L,  2.0994260633179475214672915L,  0.0750009251580063385127038L },
  {  1.4256238701470745341917051L, -2.0994260633179475214672915L,  0.0750009251580063385127038L },
  {  1.7159687834427189660703261L, -2.0994260633179475214672915L,  0.0750009251580063385127038L },
  {  0.5463708680698394719662660L,  1.2886499353585771190664034L,  0.0750009251580063385127038L },
  {  2.5952217855199542551885679L,  1.2886499353585771190664034L,  0.0750009251580063385127038L },
  {  0.5463708680698394719662660L, -1.2886499353585771190664034L,  0.0750009251580063385127038L },
  {  2.5952217855199542551885679L, -1.2886499353585771190664034L,  0.0750009251580063385127038L },
  {  0.5463708680698394719662660L,  1.8529427182312161193962400L,  0.0750009251580063385127038L },
  {  2.5952217855199542551885679L,  1.8529427182312161193962400L,  0.0750009251580063385127038L },
  {  0.5463708680698394719662660L, -1.8529427182312161193962400L,  0.0750009251580063385127038L },
  {  2.5952217855199542551885679L, -1.8529427182312161193962400L,  0.0750009251580063385127038L },
  {  1.0482995747578576810881271L,  1.4030746640453998366642749L,  0.0750009251580063385127038L },
  {  2.0932930788319356795475639L,  1.4030746640453998366642749L,  0.0750009251580063385127038L },
  {  1.0482995747578576810881271L, -1.4030746640453998366642749L,  0.0750009251580063385127038L },
  {  2.0932930788319356795475639L, -1.4030746640453998366642749L,  0.0750009251580063385127038L },
  {  1.0482995747578576810881271L,  1.7385179895443934017983685L,  0.0750009251580063385127038L },
  {  2.0932930788319356795475639L,  1.7385179895443934017983685L,  0.0750009251580063385127038L },
  {  1.0482995747578576810881271L, -1.7385179895443934017983685L,  0.0750009251580063385127038L },
  {  2.0932930788319356795475639L, -1.7385179895443934017983685L,  0.0750009251580063385127038L },
  {  1.4256238701470745341917051L,  0.5286297365230509022359698L,  0.0750009251580063385127038L },
  {  1.7159687834427189660703261L,  0.5286297365230509022359698L,  0.0750009251580063385127038L },
  {  1.4256238701470745341917051L, -0.5286297365230509022359698L,  0.0750009251580063385127038L },
  {  1.7159687834427189660703261L, -0.5286297365230509022359698L,  0.0750009251580063385127038L },
  {  1.4256238701470745341917051L,  2.6129629170667428249188641L,  0.0750009251580063385127038L },
  {  1.7159687834427189660703261L,  2.6129629170667428249188641L,  0.0750009251580063385127038L },
  {  1.4256238701470745341917051L, -2.6129629170667428249188641L,  0.0750009251580063385127038L },
  {  1.7159687834427189660703261L, -2.6129629170667428249188641L,  0.0750009251580063385127038L },
  {  1.0482995747578576810881271L,  0.1677216627494966254874141L,  0.0750009251580063385127038L },
  {  2.0932930788319356795475639L,  0.1677216627494966254874141L,  0.0750009251580063385127038L },
  {  1.0482995747578576810881271L, -0.1677216627494966254874141L,  0.0750009251580063385127038L },
  {  2.0932930788319356795475639L, -0.1677216627494966254874141L,  0.0750009251580063385127038L },
  {  1.0482995747578576810881271L,  2.9738709908402969620410796L,  0.0750009251580063385127038L },
  {  2.0932930788319356795475639L,  2.9738709908402969620410796L,  0.0750009251580063385127038L },
  {  1.0482995747578576810881271L, -2.9738709908402969620410796L,  0.0750009251580063385127038L },
  {  2.0932930788319356795475639L, -2.9738709908402969620410796L,  0.0750009251580063385127038L },
};
