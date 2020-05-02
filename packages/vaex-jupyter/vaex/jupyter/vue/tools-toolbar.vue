<template>
  <v-toolbar flat>
    <v-btn-toggle v-model="interact_value">
      <v-tooltip bottom v-for="item in interact_items" :key="item.value">
          <template v-slot:activator="{ on }">
              <v-btn v-on="on" :value="item.value">
                  <v-icon>{{ item.icon }}</v-icon>
                  <span v-if="item.html" v-html="item.html"></span>
              </v-btn>
          </template>
        <span>{{ item.tooltip }}</span>
      </v-tooltip>
    </v-btn-toggle>
    <v-select v-if="supports_transforms" v-model="transform_value" :items="transform_items" class="pa-4" style="max-width: 180px"/>
  </v-toolbar>
</template>