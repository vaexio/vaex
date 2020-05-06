<template>
  <v-card class="mx-auto" v-bind="card_props">
    <v-card-title v-show="title">{{ title }}</v-card-title>
    <v-card-subtitle v-show="subtitle">{{ subtitle }}</v-card-subtitle>
    <v-container v-bind="main_props">
      <jupyter-widget :widget="main"></jupyter-widget>
    </v-container>
    <v-card-text v-html="text">
    </v-card-text>

    <v-card-actions>
      <v-btn icon @click="show_controls = !show_controls">
        <v-icon>{{ show_controls ? 'expand_less' : 'expand_more' }}</v-icon>
      </v-btn>
    </v-card-actions>

    <v-expand-transition>
      <v-card-actions v-show="show_controls" style="display: block">
        <template>
            <v-divider></v-divider>
            <jupyter-widget v-for="(child, index) in controls" :widget="child" :key="child + '_' + index"></jupyter-widget>
        </template>
      </v-card-actions>
    </v-expand-transition>
  </v-card>
</template>

