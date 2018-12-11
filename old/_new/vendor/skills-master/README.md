# skills
Skills is a jquery plugin to create your own skill bars in your web, bars get completed while the user scroll down ,this kind of skill bars are widely used for portafolio websites just like mine portafolio.msepulveda.tk, take a look to see this plugin in action.

##  just add this html markup

 
    <section id="habilidades" class="skills">
    	<h1 class="letra_grande" align="center">skills</h1>
    	
    	<div class="habilidades_contenedor">
    		<h2>Desarrollo WEB</h2>
    		<div id="codeconSkills">
    			<div class="codeconSkillbar">
    				<div id="" class="skillBar" skill-percentage="100%" skill-color="#448AFF"> 
    					<span class="codeconSkillArea ">HTML</span>
    					<span class="PercentText ">100%</span>
    				</div>
    		      	<img src="{{ asset('img/curriculum/html_5_c.png') }}" title="HTML 5" alt="Programación en PHP" class="skill-image"  class="">
    
    			</div>
    			<div class="codeconSkillbar">
    			<div id="" class="skillBar" skill-percentage="80%" skill-color="#FFC107">
    				<span class="codeconSkillArea">Javascript</span>
    				<span class="PercentText">80%</span>
    
    			</div>
    	      		<img src="{{ asset('img/curriculum/js-flat.png') }}" alt="Javascript" class="skill-image" title="Javascript" class="">
    
    			</div>
    		<div class="codeconSkillbar">
    			<div id="" class="skillBar" skill-percentage="80%" skill-color="#009688">
    				<span class="codeconSkillArea ">CSS/CSS3</span>
    				<span class="PercentText">80%</span>
    			</div>
    	      	<img src="{{ asset('img/curriculum/css_c.png') }}" title="CSS 3" alt="Programación en PHP" class="skill-image"  class="">
    
    		</div>
    		<div class="codeconSkillbar">
    			<div id="" class="skillBar" skill-percentage="80%" skill-color="#ff5722">
    				<span class="codeconSkillArea">jQuery</span>
    				<span class="PercentText ">80%</span>
    
    			</div>
    	      	<img src="{{ asset('img/curriculum/jquery_c.png') }}" title="JQuery" alt="Programación en PHP" class="skill-image"  class="">
    
    		</div>
    		<div class="codeconSkillbar">
    			<div id="" class="skillBar" skill-percentage="70%" skill-color="#795548">
    				<span class="codeconSkillArea ">CakePHP</span>
    				<span class="PercentText">60%</span>
    
    			</div>
    	      	<img src="{{ asset('img/curriculum/cake_php.png') }}" alt="CakePHP" class="skill-image" title="CakePHP" class="">
    
    		</div>
    		<div class="codeconSkillbar">
    			<div id="" class="skillBar" skill-percentage="50%" skill-color="#C2185B">
    				<span class="codeconSkillArea">Laravel</span>
    				<span class="PercentText">50%</span>
    
    			</div>
    	      	<img src="{{ asset('img/curriculum/laravel_c.png') }}" alt="Programación en PHP" class="skill-image" title="Laravel" class="">
    
    		</div>
    		<div class="codeconSkillbar">
    			<div id="" class="skillBar" skill-percentage="90%" skill-color="#673AB7">
    				<span class="codeconSkillArea">PHP</span>
    				<span class="PercentText">90%</span>
    
    			</div>
    	      	<img src="{{ asset('img/curriculum/php_icon_c.png') }}" alt="Programación en PHP" class="skill-image" title="PHP" class="">
    
    		</div>
    		<div class="codeconSkillbar">
    			<div id="" class="skillBar" skill-percentage="40%" skill-color="#FF5252">
    				<span class="codeconSkillArea">Rails</span>
    				<span class="PercentText">40%</span>
    
    			</div>
    	      	<img src="{{ asset('img/curriculum/rails_c.png') }}" alt="Rails" class="skill-image" title="Rails" class="">
    
    		</div>
    		<div class="codeconSkillbar">
    			<div id="" class="skillBar" skill-percentage="60%" skill-color="#D32F2F">
    				<span class="codeconSkillArea">Angular</span>
    				<span class="PercentText">60%</span>
    
    			</div>
    	      	<img src="{{ asset('img/curriculum/angular_c.png') }}" alt="Angular" class="skill-image" title="Angular" class="">
    
    		</div>
    		
    	</div>
    	</div>
    
    </section>
	    

### Then, call the necesary files in your head tag:


    <link rel="stylesheet" href="source/animate.css">
  	<link rel="stylesheet" href="source/habilidades.css">
  	<script src="https://ajax.googleapis.com/ajax/libs/jquery/1.12.4/jquery.min.js"></script>
  	<script src="jquery.skills.js"></script>
  	
### Finally initialize the plugin
  	    <script>
      		$(document).ready(function() {
      			$(".skills").skill();
      		});
    	</script>
