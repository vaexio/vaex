$.fn.skill = function() {

	mSkill = this;
  $(window).on('scroll', function() { // las animaciones se dispararan cuando el recuadro este visible en el viewport
	
	mSkill.find('.skillBar').each(function() {

		if( $(this).offset().top <= $(window).scrollTop()+$(window).height()*0.90 &&!$(this).hasClass("sk-fired")) {
			//una vez que cada skill bar esta en el viewport
			
			$(this).addClass('sk-fired'); //agregamos una clase como bandera para evitar que se vuelva a reproducir la animacion
			var defaultPercentage = "50%";
			//animamos el ancho de cada barra
			if($(this).attr('data-percent')) {
				$(this).width($(this).attr('data-percent'));
			} else {
				$(this).width(defaultPercentage);
			}


			//buscamos las imagenes para animarlas
			$(this).parent().find(".skill-image").each(function() {
				var imagen = $(this);
				setInterval(function() {

					imagen.show().addClass("animated").addClass("bounceIn");
				}, 2000);
				
			});
			}
		});

}	);

     return mSkill;
}