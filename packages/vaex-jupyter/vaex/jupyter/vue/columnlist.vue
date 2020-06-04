<template>
  <v-layout column>
    <v-dialog v-model="dialog_open" width="500" @keydown.stop>
      <v-card>
        <v-card-title class="headline grey lighten-2" primary-title>Add a virtual column</v-card-title>
        <content-editor></content-editor>
        <v-divider></v-divider>
        <v-card-actions>
          <v-spacer></v-spacer>
          <v-btn color="secondary" text @click="dialog_open = false">Cancel</v-btn>
          <v-btn
            color="primary"
            text
            @click="save_column"
            :disabled="!valid_expression"
          >Add/save column</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
    <v-subheader class="grey lighten-4" style="min-height: 40px; position: relative" key="header">
      <v-menu offset-y>
        <template v-slot:activator="{ on: menu }">
          <v-btn absolute bottom right color="primary" v-on="{...menu}" fab small>
            <v-icon>add</v-icon>
          </v-btn>
        </template>
        <v-list>
          <v-list-item @click="add_virtual_column()">
            <v-list-item-title>Add virtual column</v-list-item-title>
          </v-list-item>
        </v-list>
      </v-menu>Columns
    </v-subheader>
    <v-text-field
      class="ma-3"
      label="Filter column names"
      placeholder="e.g. name"
      v-model="column_filter"
      key="filter"
    />
    <v-scroll-y-transition tag="v-list" group leave-absolute class="overflow-y-auto">
      <v-list-item
        v-if="!column_filter || column.name.includes(column_filter)"
        v-for="(column, index) in columns"
        @click="column_click(column)"
        :key="index"
      >
        <v-list-item-avatar>
          <v-icon class="grey white--text" v-if="!column.virtual">view_column</v-icon>
          <v-icon class="grey white--text" v-else>functions</v-icon>
        </v-list-item-avatar>
        <v-list-item-content>
          <v-list-item-title>{{ column.name }}</v-list-item-title>
          <v-list-item-subtitle v-html="column.dtype"></v-list-item-subtitle>
        </v-list-item-content>
      </v-list-item>
    </v-scroll-y-transition>
  </v-layout>
</template>