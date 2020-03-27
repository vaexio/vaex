<template>
  <div>
    <v-speed-dial
        style="bottom: 80px"
        absolute
        v-model="expand"
        bottom
        right
        direction="top"
        transition="slide-y-reverse-transition"
    >
        <template v-slot:activator>
        <v-btn v-model="expand" small fab>
            <v-icon v-if="expand">expand_more</v-icon>
            <v-icon v-else>more_vert</v-icon>
        </v-btn>
        </template>
        <v-tooltip v-for="(item, index) in items" left :key="index">
        <template v-slot:activator="tooltip">
            <v-btn fab small v-on="tooltip.on" @click="action(item)">
            <v-icon v-if="item.icon" small>{{ item.icon }}</v-icon>
            <span v-if="item.html" v-html="item.html"></span>
            </v-btn>
        </template>
        {{ item.tooltip }}
        </v-tooltip>
      </v-speed-dial>
      <jupyter-widget v-for="(child, index) in children" :widget="child" :key="child + '_' + index">
      </jupyter-widget>
  </div>
</template>