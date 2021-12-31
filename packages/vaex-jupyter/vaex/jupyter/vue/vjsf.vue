<template>
  <div>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@koumoul/vjsf@latest/dist/main.css">

    <div v-if="vjsf_loaded" style="padding: 0px">
      <v-form ref="form" v-model="valid">
        <v-jsf v-model="values" :schema="schema"></v-jsf>
      </v-form>
      <v-layout row class="ma-4">
        <v-btn color="primary" :disabled="!valid" @click="$refs.form.validate()">Validate</v-btn>
    </v-layout> 
    </div>
  </div>    
</template>

<script>
module.exports = {
    async created() {
      this.vjsf_loaded = false
        console.log('created')
        const [VJsf] = await this.import(['https://cdn.jsdelivr.net/npm/@koumoul/vjsf@latest/dist/main.js']);
        this.$options.components['v-jsf'] = VJsf.default;
        this.vjsf_loaded = true;
    },
    methods: {
        import(deps) {
          return this.loadRequire()
              .then(() => new Promise((resolve, reject) => {
                requirejs(deps, (...modules) => resolve(modules));
              }));
        },
        loadRequire() {
          /* Needed in lab */
          if (window.requirejs) {
              console.log('require found');
              return Promise.resolve()
          }
          return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = 'https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js';
            script.onload = resolve;
            script.onerror = reject;
            document.head.appendChild(script);
          });
        }
    }
}
</script>