_obj-to-html_ is a small application that converts a 3D model in .obj format into HTML and CSS that will display that model in a web browser, spinning around its Y axis. This is achieved through creative abuse of CSS's 3D transforms. It supports triangle-based models and can pick up diffuse colours and textures from a .mtl file. I've tested it on a handful of models from [Models Resource](https://www.models-resource.com/).

This is absolutely not the best way to display a 3D model in the browser — please use WebGL for serious projects! But it is quite fun to use on websites like [cohost](https://cohost.org) that let you make posts containing arbitrary inline CSS, but not inline JS. For example, [this post where I show off Lara Croft from Tomb Raider](https://cohost.org/hikari-no-yume/post/459729-lara-croft-from-tomb) (and announce this project!)

This was written by me over three days (24th to 26th November 2022). Special thanks to [cassie](https://www.witchoflight.com/) who listened to my rambling about the project and made some helpful suggestions. ^^

In January 2023 [I wrote a blog post about the project](https://hikari.noyu.me/blog/2023-01-10-polygons-from-paragraphs-3d-model-html-css.html), if you want to know a bit about the process.
